import argparse
import traceback
from pathlib import Path
import sys
import os
from platform import platform

from mochi import __version__, IS_PYPY, GE_PYTHON_34, GE_PYTHON_33
from .builtins import current_error_port, eval_sexp_str, eval_tokens
from mochi.parser import lex, REPL_CONTINUE
from .global_env import global_env
from .translation import syntax_table, global_scope, translator


def output_code(code):
    import marshal

    marshal.dump(code, sys.stdout.buffer)


def output_pyc(code, buffer=sys.stdout.buffer):
    import marshal

    if GE_PYTHON_34:
        from importlib.util import MAGIC_NUMBER
    else:
        import imp
        MAGIC_NUMBER = imp.get_magic()

    buffer.write(MAGIC_NUMBER)
    if GE_PYTHON_33:
        buffer.write(b'0' * 8)
    else:
        buffer.write(b'0' * 4)
    marshal.dump(code, buffer)
    buffer.flush()


def compile_file(src_path, optimize=-1, show_tokens=False):
    # binding_name_set_stack[0].update(global_env.keys())
    py_ast = translator.translate_file(src_path, show_tokens=show_tokens)
    sys.modules['__main__'] = global_env
    return compile(py_ast, src_path, 'exec', optimize=optimize)


def load_file(path, env):
    env['__name__'] = '__main__'
    sys.modules['__main__'] = env
    return exec(compile_file(path), env)


def execute_compiled_file(path):
    import marshal
    global_env['__name__'] = '__main__'
    sys.modules['__main__'] = global_env
    with open(path, 'rb') as compiled_file:
        return exec(marshal.load(compiled_file), global_env)


def interact(show_tokens=False):
    try:
        import readline
    except ImportError:
        pass

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
        eval_tokens(tokens)


def init():
    import eventlet

    def eval_from_file(path_obj):
        """Evaluate sexpression in given file.
        """
        with path_obj.open() as fobj:
            expr = fobj.read()
        eval_sexp_str(expr)

    if platform().lower().startswith('win'):
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
    env='import_global_env_and_monkey_patch.mochi'
    _pyc_compile(in_file_name, env, out_file_name, show_tokens=show_tokens)


def pyc_compile_no_monkeypatch(in_file_name, out_file_name=None, show_tokens=False):
    env='import_global_env.mochi'
    _pyc_compile(in_file_name, env, out_file_name, show_tokens=show_tokens)


def parse_args():
    arg_parser = argparse.ArgumentParser(
        description='Mochi is a functional programming language.')
    arg_parser.add_argument('-v', '--version', action='version',
                            version=__version__)
    arg_parser.add_argument('-c', '--compile', action='store_true')
    arg_parser.add_argument('-pyc', '--pyc-compile', action='store_true')
    arg_parser.add_argument('-pyc-no-mp', '--pyc-compile-no-monkeypatch',
                            action='store_true')
    arg_parser.add_argument('-e', '--execute-compiled-file',
                            action='store_true')
    arg_parser.add_argument('file', nargs='?', type=str)
    arg_parser.add_argument('--show-tokens', dest='tokens', help='Shows the results of the tokenizing step', action='store_true')

    return arg_parser.parse_args()


def main():
    args = parse_args()
    if args.file:
        if args.compile:
            output_code(compile_file(args.file, optimize=2, show_tokens=args.tokens))
        elif args.execute_compiled_file:
            execute_compiled_file(args.file)
        elif args.pyc_compile:
            pyc_compile_monkeypatch(in_file_name=args.file, show_tokens=args.tokens)
        elif args.pyc_compile_no_monkeypatch:
            pyc_compile_no_monkeypatch(in_file_name=args.file, show_tokens=args.tokens)
        else:
            load_file(args.file, global_env)
        sys.exit(0)
    else:
        interact(args.tokens)

init()

if __name__ == '__main__':
    main()
