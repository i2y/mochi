import argparse
from pathlib import Path
import sys
import os
from platform import platform, system
import traceback

import astunparse

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
    buffer.write(timestamp)
    if GE_PYTHON_33:
        buffer.write(b'0' * 4)
    marshal.dump(code, buffer)
    buffer.flush()


def compile_file(src_path, optimize=-1, show_tokens=False):
    # binding_name_set_stack[0].update(global_env.keys())
    py_ast = translator.translate_file(src_path, show_tokens=show_tokens)
    return compile(py_ast, src_path, 'exec', optimize=optimize)


def load_file(path, env, optimize=-1):
    return exec(compile_file(path, optimize), env)


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


def _pyc_compile(in_file_name, env, out_file_name, show_tokens=False, optimize=-1):
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
    output_pyc(compile(target_ast, in_file_name, 'exec', optimize=optimize),
               buffer=out_file)


def pyc_compile_monkeypatch(in_file_name, out_file_name=None, show_tokens=False, optimize=-1):
    env = 'import_global_env_and_monkey_patch.mochi'
    _pyc_compile(in_file_name, env, out_file_name, show_tokens=show_tokens, optimize=optimize)


def pyc_compile_no_monkeypatch(in_file_name, out_file_name=None, show_tokens=False, optimize=-1):
    env = 'import_global_env.mochi'
    _pyc_compile(in_file_name, env, out_file_name, show_tokens=show_tokens, optimize=optimize)


def make_py_source_file(mochi_file_name, python_file_name=None, mochi_env='',
                        add_init=False, show_tokens=False):
    """Generate Python source code from Mochi code.
    """
    ast = translator.translate_file(mochi_file_name, show_tokens=show_tokens)
    if mochi_env:
        env_file = Path(__file__).absolute().parents[0] / mochi_env
        with open(env_file.as_posix()) as fobj:
            mochi_env = fobj.read()
    py_source = clean_source(ast2py(ast, mochi_env, add_init=add_init))
    if not python_file_name:
        print(py_source)
    else:
        with open(python_file_name, 'w') as fobj:
            fobj.write(py_source)


def pprint_ast(mochi_file_name, ast_file_name=None, show_tokens=False):
    """Generate a nicly formatted AST from Mochi code.
    """
    ast = translator.translate_file(mochi_file_name, show_tokens=show_tokens)
    py_source = astunparse.dump(ast)
    if not ast_file_name:
        print(py_source)
    else:
        with open(ast_file_name, 'w') as fobj:
            fobj.write(py_source)


def clean_source(source):
    # TODO: Fix AST generation so this function is not needed.
    """Dirty cleaning of dirty source."""
    # replace '$_x'  with 'arg_x' x = 1, 2, 3 ... 9
    if '$' in source:
        for number in range(1, 10):
            source = source.replace('${}'.format(number),
                                            'arg_{}'.format(number))
    # remove extra `try` with no use but messing up syntax
    if 'try' in source:
        lines = source.splitlines()
        new_lines = [line for line in lines if not line.strip() == 'try']
        source = '\n'.join(new_lines)
    if '|>(' in source:
        source = source.replace('|>(', 'bind(')
    val = 'val('
    if val in source:
        lines = source.splitlines()
        new_lines = []
        for line in lines:
            if line.strip().startswith(val):
                spaces = int(line.index(val)) * ' '
                name, value = line.split(val)[1].split(',', 1)
                assign = '{} = {}'.format(name, value[:-1])
                new_lines.append(spaces + assign)
            else:
                new_lines.append(line)
        source = '\n'.join(new_lines)
    # TODO: fix `&
    #if '&' in source:
    #    source = source.replace('&', '*_rest')
    return source


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
    arg_parser.add_argument('-a', '--ast', action='store_true',
                            help='Generate AST from Mochi file.')
    arg_parser.add_argument('-o', '--outfile', nargs='?', type=str,
                            help='Name of output file.')
    arg_parser.add_argument('-O', '--optimize', action='store_true',
                            help='Optimize generated bytecode slightly.')
    arg_parser.add_argument('-OO', '--optimize2', action='store_true',
                            help='Remove doc-strings in addition to the -O optimizations.')
    arg_parser.add_argument('-no-mp', '--no-monkeypatch',
                            action='store_true')
    arg_parser.add_argument('-mc', '--multi-core', action='store_true',
                            help='''Add spawn_mc function that executes actors in multiple processes
                             to gain effects of multi-core processor'''),
    arg_parser.add_argument('-init', '--add-init-code', action='store_true',
                            help='Add Mochi init code to Python source code '
                                 'files. This allows running the generated '
                                 'file from the command line with Python.')
    arg_parser.add_argument('-e', '--execute-compiled-file',
                            action='store_true')
    arg_parser.add_argument('file', nargs='?', type=str)
    arg_parser.add_argument('args', nargs='*', type=str)
    arg_parser.add_argument('--show-tokens', dest='tokens',
                            help='Shows the results of the tokenizing step.',
                            action='store_true')

    return arg_parser.parse_args()


def run_processes(cpu_nums):
    from multiprocessing import Process
    from mochi.actor.actor import spawn_with_mailbox, make_ref
    from mochi.actor.mailbox import IpcInbox, IpcInboxR

    def processor(address):
        inbox = IpcInboxR(address)
        while True:
            message = inbox.get()
            fun, args, actor_address = message
            spawn_with_mailbox(fun,
                               IpcInbox(actor_address),
                               *args)

    address = str(make_ref())
    for i in range(cpu_nums):
        p = Process(target=processor, args=(address,))
        p.start()
        if 'Linux' in system():
            os.system('taskset -p -c %d %d' % ((i % cpu_nums), p.pid))
    return address


def main():
    args = parse_args()
    init(args.no_monkeypatch)
    if args.file:
        try:
            if args.no_monkeypatch:
                env = GLOBAL_ENV
            else:
                env = MONKEY_PATCH_ENV

            if not (args.optimize or args.optimize2):
                optimize = 0
            elif args.optimize:
                optimize = 1
            else:
                optimize = 2

            if args.compile:
                output_code(compile_file(args.file,
                                         optimize,
                                         show_tokens=args.tokens))
            elif args.execute_compiled_file:
                execute_compiled_file(args.file)
            elif args.pyc_compile:
                if args.no_monkeypatch:
                    pyc_compile_no_monkeypatch(in_file_name=args.file,
                                               out_file_name=args.outfile,
                                               show_tokens=args.tokens,
                                               optimize=optimize)
                else:
                    pyc_compile_monkeypatch(in_file_name=args.file,
                                            out_file_name=args.outfile,
                                            show_tokens=args.tokens,
                                            optimize=optimize)
            elif args.py_source:
                make_py_source_file(mochi_file_name=args.file,
                                    python_file_name=args.outfile,
                                    mochi_env=env,
                                    show_tokens=args.tokens,
                                    add_init=args.add_init_code)
            elif args.ast:
                pprint_ast(mochi_file_name=args.file,
                           ast_file_name=args.outfile, show_tokens=args.tokens)
            else:
                sys.modules['__main__'] = global_env

                if args.multi_core:
                    address = run_processes(os.cpu_count())
                    from mochi.actor.actor import make_ref
                    from mochi.actor.mailbox import IpcOutbox, IpcInbox, IpcOutboxR
                    from itertools import cycle
                    from eventlet.green import zmq

                    outbox = IpcOutboxR(address)

                    def spawn_m(fun, *args):
                        new_address = str(make_ref())
                        outbox.put([fun, args, new_address])
                        return IpcOutbox(new_address)
                    global_env['spawn_mc'] = spawn_m

                load_file(args.file, global_env, optimize=optimize)
        except ParsingError as e:
                print(e, file=sys.stderr)
        except Exception:
                traceback.print_exc(file=sys.stderr)
        sys.exit(0)
    else:
        interact(args.tokens)


if __name__ == '__main__':
    main()
