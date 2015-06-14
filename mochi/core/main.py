import argparse
from pathlib import Path
import sys
import os
from platform import platform
import traceback

from mochi import __version__, IS_PYPY, GE_PYTHON_34, GE_PYTHON_33
from mochi.parser import lex, REPL_CONTINUE, ParsingError
from .builtins import current_error_port, eval_sexp_str, eval_tokens
from .global_env import global_env
from .translation import syntax_table, global_scope, translator, ast2py


MONKEY_PATCH_ENV = 'import_global_env_and_monkey_patch.mochi'
GLOBAL_ENV = 'import_global_env.mochi'


def output_code(code):
    import marshal

    marshal.dump(code, sys.stdout.buffer)


def output_pyc(code, buffer=sys.stdout.buffer):
    import marshal
    import struct
    import time

    if GE_PYTHON_34:
        from importlib.util import MAGIC_NUMBER
    else:
        import imp
        MAGIC_NUMBER = imp.get_magic()

    buffer.write(MAGIC_NUMBER)
    timestamp = struct.pack('i', int(time.time()))
    if GE_PYTHON_33:
        buffer.write(timestamp)
        buffer.write(b'0' * 4)
    else:
        buffer.write(timestamp)
    marshal.dump(code, buffer)
    buffer.flush()


def compile_file(src_path, optimize=-1, show_tokens=False):
    # binding_name_set_stack[0].update(global_env.keys())
    py_ast = translator.translate_file(src_path, show_tokens=show_tokens)
    return compile(py_ast, src_path, 'exec', optimize=optimize)


def load_file(path, env):
    return exec(compile_file(path), env)


def execute_compiled_file(path):
    import marshal
    orig_main = sys.modules['__main__']
    sys.modules['__main__'] = global_env
    try:
        with open(path, 'rb') as compiled_file:
            return exec(marshal.load(compiled_file), global_env)
    finally:
        sys.modules['__main__'] = orig_main


def interact(show_tokens=False):
    try:
        import readline
    except ImportError:
        pass

    sys.modules['__main__'] = global_env

    while True:
        buffer = ''
        continuation_flag = False
        tokens = []
        while True:
            try:
                if continuation_flag:
                    s = input('... ')
                    if s == '\n':
                        continue
                    buffer = buffer + '\n' + s
                else:
                    s = input('>>> ')
                    if s == '\n':
                        continue
                    buffer = s
            except EOFError:
                print()
                sys.exit()

            try:
                lexer = lex(buffer, repl_mode=True, debug=show_tokens)

                for last in lexer:
                    tokens.append(last)

                if len(tokens) == 0:
                    buffer = ''
                    continue

                if last is REPL_CONTINUE or last.name == 'COLON' or last.name == 'THINARROW':
                    continuation_flag = True
                    tokens = []
                    continue
                else:
                    break
            except Exception:
                traceback.print_exc(file=current_error_port)
                continuation_flag = False
                buffer = ''
                continue
        try:
            eval_tokens(tokens)
        except ParsingError as e:
            print(e, file=current_error_port)
        except Exception:
            traceback.print_exc(file=current_error_port)


def init(no_monkeypatch=False):
    if hasattr(init, '__called') and init.__called:
        return
    else:
        init.__called = True

    import eventlet

    def eval_from_file(path_obj):
        """Evaluate sexpression in given file.
        """
        with path_obj.open() as fobj:
            expr = fobj.read()
        eval_sexp_str(expr)

    if no_monkeypatch:
        pass
    elif (not GE_PYTHON_33) or platform().lower().startswith('win'):
        eventlet.monkey_patch(os=False)
    else:
        eventlet.monkey_patch()
    expr_path = Path(__file__).absolute().parents[1] / 'sexpressions'
    eval_from_file(expr_path / 'main.expr')
    if not IS_PYPY:
        eval_from_file(expr_path / 'cpython.expr')
    else:
        eval_from_file(expr_path / 'pypy.expr')

    eval_from_file(expr_path / 'del_hidden.expr')

    for syntax in {'for', 'each', 'while', 'break', 'continue'}:
        del syntax_table[syntax]
        del global_env[syntax]
        del global_scope[syntax]

    sys.path.append(os.getcwd())
    from mochi.utils.importer import set_importer
    set_importer()


def _pyc_compile(in_file_name, env, out_file_name, show_tokens=False):
    """Compile a Mochi file into a Python bytecode file.
    """
    if not out_file_name:
        out_file = sys.stdout.buffer
    else:
        out_file = open(out_file_name, 'wb')
    target_ast = translator.translate_file(in_file_name, show_tokens=show_tokens)
    import_env_file = Path(__file__).absolute().parents[0] / env
    import_env_ast = translator.translate_file(import_env_file.as_posix())
    target_ast.body = import_env_ast.body + target_ast.body
    output_pyc(compile(target_ast, in_file_name, 'exec', optimize=2),
               buffer=out_file)


def pyc_compile_monkeypatch(in_file_name, out_file_name=None, show_tokens=False):
    env = 'import_global_env_and_monkey_patch.mochi'
    _pyc_compile(in_file_name, env, out_file_name, show_tokens=show_tokens)


def pyc_compile_no_monkeypatch(in_file_name, out_file_name=None, show_tokens=False):
    env = 'import_global_env.mochi'
    _pyc_compile(in_file_name, env, out_file_name, show_tokens=show_tokens)


def make_py_source_file(mochi_file_name, python_file_name=None, mochi_env='',
                        add_init=False, show_tokens=False):
    """Generate Python source code from Mochi code.
    """
    ast = translator.translate_file(mochi_file_name, show_tokens=show_tokens)
    if mochi_env:
        env_file = Path(__file__).absolute().parents[0] / mochi_env
        with open(env_file.as_posix()) as fobj:
            mochi_env = fobj.read()
    py_source = ast2py(ast, mochi_env, add_init=add_init)
    if not python_file_name:
        print(py_source)
    else:
        with open(python_file_name, 'w') as fobj:
            fobj.write(py_source)


def parse_args():
    arg_parser = argparse.ArgumentParser(
        description='Mochi is a functional programming language.')
    arg_parser.add_argument('-v', '--version', action='version',
                            version=__version__)
    arg_parser.add_argument('-c', '--compile', action='store_true',
                            help='Show marshalled code.')
    arg_parser.add_argument('-pyc', '--pyc-compile', action='store_true',
                            help='Generate Python bytecode from Mochi file.')
    arg_parser.add_argument('-py', '--py-source', action='store_true',
                            help='Generate Python source code from Mochi file.')
    arg_parser.add_argument('-o', '--outfile', nargs='?', type=str,
                            help='Name of output file.')
    arg_parser.add_argument('-no-mp', '--no-monkeypatch',
                            action='store_true')
    arg_parser.add_argument('-init', '--add-init-code', action='store_true',
                            help='Add Mochi init code to Python source code '
                                 'files. This allows running the from the '
                                 'command line with Python.')
    arg_parser.add_argument('-e', '--execute-compiled-file',
                            action='store_true')
    arg_parser.add_argument('file', nargs='?', type=str)
    arg_parser.add_argument('--show-tokens', dest='tokens',
                            help='Shows the results of the tokenizing step.',
                            action='store_true')

    return arg_parser.parse_args()


def main():
    args = parse_args()
    init(args.no_monkeypatch)
    if args.file:
        try:
            if args.no_monkeypatch:
                env = GLOBAL_ENV
            else:
                env = MONKEY_PATCH_ENV
            if args.compile:
                output_code(compile_file(args.file,
                                         optimize=2,
                                         show_tokens=args.tokens))
            elif args.execute_compiled_file:
                execute_compiled_file(args.file)
            elif args.pyc_compile:
                if args.no_monkeypatch:
                    pyc_compile_no_monkeypatch(in_file_name=args.file,
                                               show_tokens=args.tokens)
                else:
                    pyc_compile_monkeypatch(in_file_name=args.file,
                                            show_tokens=args.tokens)
            elif args.py_source:
                make_py_source_file(mochi_file_name=args.file,
                                    python_file_name=args.outfile,
                                    mochi_env=env,
                                    show_tokens=args.tokens,
                                    add_init=args.add_init_code)
            else:
                sys.modules['__main__'] = global_env
                load_file(args.file, global_env)
        except ParsingError as e:
                print(e, file=sys.stderr)
        except Exception:
                traceback.print_exc(file=sys.stderr)
        sys.exit(0)
    else:
        interact(args.tokens)


if __name__ == '__main__':
    main()
