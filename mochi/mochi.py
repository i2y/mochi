#!/usr/bin/env python
#
# Copyright (c) 2014 Yasushi Itoh
#
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php

import ast
import sys
from types import FunctionType
from functools import reduce, partial
from itertools import chain
from operator import concat, add
from collections import Iterator, Iterable, Mapping, MutableMapping, Sequence, MutableSequence, namedtuple
from numbers import Number
import argparse
from io import StringIO
from pathlib import Path
from os import chdir
from os.path import normpath, abspath
import platform

from pyrsistent import v, pvector, m, pmap, s, pset, b, pbag, dq, pdeque, l, plist, pclass, freeze, thaw

# import traceback

from mochi.parser import Symbol, Keyword, parse, lex, REPL_CONTINUE


IS_PYTHON_34 = sys.version_info.major == 3 and sys.version_info.minor == 4
IS_PYPY = platform.python_implementation() == 'PyPy'

if IS_PYTHON_34:
    def wr_long(f, x):
        """Internal; write a 32-bit int to a file in little-endian order."""
        f.write(bytes([x & 0xff,
                       (x >> 8) & 0xff,
                       (x >> 16) & 0xff,
                       (x >> 24) & 0xff]))

    from importlib.util import MAGIC_NUMBER as MAGIC
else:
    from py_compile import wr_long, MAGIC


def issequence(obj):
    return isinstance(obj, Sequence)


def issequence_except_str(obj):
    if isinstance(obj, str):
        return False
    return isinstance(obj, Sequence)


def is_tuple_or_list(obj):
    return type(obj) in {tuple, list}


def make_default_env():
    import functools
    import itertools
    import operator
    import re

    env = {'Symbol': Symbol, 'Keyword': Keyword}
    env.update(__builtins__.__dict__) if hasattr(__builtins__, '__dict__') else env.update(__builtins__)
    del env['exec']
    # del env['globals']
    # del env['locals']
    env.update(functools.__dict__)
    env.update(itertools.__dict__)
    env.update(operator.__dict__)
    env[Iterable.__name__] = Iterable
    env[Sequence.__name__] = Sequence
    env[Mapping.__name__] = Mapping
    env['v'] = v
    env['pvector'] = pvector
    env['m'] = m
    env['pmap'] = pmap
    env['s'] = s
    env['pset'] = pset
    env['l'] = l
    env['plist'] = plist
    env['b'] = b
    env['pbag'] = pbag
    env['dq'] = dq
    env['pdeque'] = pdeque
    env['thaw'] = thaw
    env['freeze'] = freeze
    env['pclass'] = pclass
    env[Number.__name__] = Number
    env['append'] = MutableSequence.append
    # env['clear'] = MutableSequence.clear # not supported (pypy)
    env['seq-count'] = MutableSequence.count
    env['extend'] = MutableSequence.extend
    env['insert'] = MutableSequence.insert
    env['pop'] = MutableSequence.pop
    env['remove'] = MutableSequence.remove
    env['reverse'] = MutableSequence.reverse
    env['mapping-get'] = MutableMapping.get
    env['items'] = MutableMapping.items
    env['values'] = MutableMapping.values
    env['keys'] = MutableMapping.keys
    env['mapping-pop'] = MutableMapping.pop
    env['popitem'] = MutableMapping.popitem
    env['setdefault'] = MutableMapping.setdefault
    env['update'] = MutableMapping.update
    env['values'] = MutableMapping.values
    env['doall'] = pvector
    env['nth'] = operator.getitem
    env['+'] = operator.add
    env['-'] = operator.sub
    env['/'] = operator.truediv
    env['*'] = operator.mul
    env['%'] = operator.mod
    env['**'] = operator.pow
    env['<<'] = operator.lshift
    env['>>'] = operator.rshift
    env['//'] = operator.floordiv
    env['=='] = operator.eq
    env['!='] = operator.ne
    env['>'] = operator.gt
    env['>='] = operator.ge
    env['<'] = operator.lt
    env['<='] = operator.le
    env['not'] = operator.not_
    env['and'] = operator.and_
    env['or'] = operator.or_
    env['is'] = operator.is_
    env['isnt'] = operator.is_not
    env['re'] = re
    env['True'] = True
    env['False'] = False
    env['None'] = None
    env['Record'] = pclass((), 'Record')  # namedtuple('Record', ())
    env['__name__'] = '__main__'
    try:
        env['__loader__'] = __loader__
    except:
        pass
    env['__package__'] = __package__
    env['__doc__'] = __doc__
    if IS_PYPY:
        env['__builtins__'] = {}
        from _continuation import continulet

        env['continulet'] = continulet
    else:
        env['__builtins__'] = {'__build_class__': __build_class__,
                               '__import__': __import__}
    return env


global_env = make_default_env()

name_seq = 0


def builtin(func):
    global_env[func.__name__] = func
    return func


def builtin_rename(new_name):
    def _builtin(func):
        global_env[new_name] = func
        return func

    return _builtin


@builtin_rename('uniq')
@builtin_rename('gensym')
def get_temp_name():
    global name_seq
    name_seq += 1
    name_symbol = Symbol('_gs%s' % name_seq)
    return name_symbol


@builtin
def flip(func):
    return lambda arg0, arg1, *rest: func(arg1, arg0, *rest)


@builtin
def mapchain(func, iterable):
    return reduce(chain, map(func, iterable))


@builtin
def mapcat(func, iterable):
    return reduce(concat, map(func, iterable))


@builtin
def mapadd(func, iterable):
    return reduce(add, map(func, iterable))


@builtin_rename('on-err')
def handle(onerror, func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        return onerror(e)


@builtin
def protect(during_func, after_func):
    try:
        return during_func()
    finally:
        after_func()


@builtin
def err(message, exc_type=Exception):
    raise exc_type(message)


@builtin
def details(exc):
    return str(exc)


@builtin
def ignore(func, default, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except:
        return default


@builtin
def silent(func, *args, **kwargs):
    return ignore(func, None, *args, **kwargs)


@builtin_rename('gen-with')
def gen_with(context, func):
    with context:
        retval = func(context)
        if isinstance(retval, Iterable):
            # yield from retval
            for value in retval:
                yield value
        else:
            yield retval


@builtin_rename('with')
def _with(context, func):
    with context:
        return func(context)


@builtin
def gen(func, iterable):
    while True:
        yield func(next(iterable))


@builtin_rename('each-with-index')
def each_with_index(iterable, func):
    for index, value in enumerate(iterable):
        func(index, value)


@builtin
def dorun(iterable):
    retval = None
    for value in iterable:
        retval = value
    return retval


@builtin
def require_py_module(*target_symbols):
    return _require_py_module(target_symbols)


@builtin
def _require_py_module(target_symbols):
    binding_name_set = binding_name_set_stack[0]
    for target_symbol in target_symbols:
        target_obj = None
        target = target_symbol.name.split('.')
        if len(target) > 1:
            package = target[0]
            module = '.'.join(target[:-1])
            target_name = target[-1]
            if target_name in binding_name_set:
                continue
            target_obj = getattr(__import__(module, fromlist=[package]), target_name)
        elif len(target) == 1:
            target_name = target[0]
            if target_name == 'builtins' or target_name in binding_name_set:
                continue
            target_obj = __import__(target_name)
        binding_name_set.add_binding_name(target_name, "<string>")  # TODO
        global_env[target_name] = target_obj
    return True


@builtin
def cons(head, tail):
    if isinstance(tail, Iterator):
        return chain((head,), tail)
    if isinstance(tail, cell):
        return cell(head, tail)
    if tail is None:
        return cell(head, None)
    return (head,) + tail


@builtin
def conj(seq, item):
    assert issequence_except_str(seq)
    return seq + type(seq)((item,))


@builtin
@builtin_rename('first')
def car(seq):
    return seq[0]


@builtin
@builtin_rename('rest')
def cdr(seq):
    return seq[1:]


@builtin
def last(seq):
    return seq[-1]


@builtin
def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def symbolp(obj):
    return type(obj) is Symbol


def symbol_to_string(sym):
    return sym.name


def string_to_symbol(str):
    return Symbol(str)


def to_obj(tkn, lineno):
    try:
        return int(tkn)
    except:
        pass
    try:
        return float(tkn)
    except:
        pass
    try:
        return complex(tkn)
    except:
        pass

    if tkn[0] == ':':
        return Keyword(tkn[1:], lineno)
    else:
        return Symbol(tkn, lineno)


# -----------------------------------------------
# EOF Class
class Eof(object):
    def __repr__(self):
        return "EOF"

    def __str__(self):
        return "EOF"

#-----------------------------------------------
# constants
EOF = Eof()  # orignal: EOF = "EOF"
QUOTE = Symbol('quote')
QUASIQUOTE = Symbol('quasiquote')
UNQUOTE = Symbol('unquote')
UNQUOTE_SPLICING = Symbol('unquote-splicing')
SPLICING = Symbol('splicing')
VARG = Symbol('&')
VKWARG = Symbol('&&')
VAL = Symbol('val')
GET = Symbol('get')
FN = Symbol('fn')
ARGS = Symbol('args')
UNDERSCORE = Symbol('_')
LEN = Symbol('len')
IF = Symbol('if')
ELSE = Symbol('else')
LPARA = Symbol('(')
RPARA = Symbol(')')
RPARA = Symbol(')')
NONE_SYM = Symbol('None')
EMPTY = ()
EMPTY_SYM = Symbol('EMPTY')
TABLE = Symbol('table')
PMAP = Symbol('pmap')
DEF = Symbol('def')
MAKE_TUPLE = Symbol('make-tuple')
MAKE_LIST = Symbol('make-list')
MAKE_DICT = Symbol('dict*')
WITH_DECORATOR = Symbol('with-decorator')
RE_COMPILE = Symbol('re.compile')
GET_REF = Symbol('get!')
RECORD_SYM = Symbol('Record')
OBJECT_SYM = Symbol('object')
global_env['EMPTY'] = EMPTY
#-----------------------------------------------


class Comment(object):
    def __str__(self):
        return "comment"

    def __repr__(self):
        return "comment"


COMMENT = Comment()


class ReadError(Exception):
    def __init__(self, file, line, nest_level):
        if nest_level > 0:
            self.msg = 'read error: "' + file + \
                       '":line ' + str(line) + ': EOF inside a list'
        else:
            self.msg = 'read error: "' + file + \
                       '":line ' + str(line) + ': extra close parenthesis'

    def __str__(self):
        return self.msg

    def __repr__(self):
        return self.msg


class Char(object):
    def __init__(self, str, lineno=0):
        self.value = str
        self.lineno = lineno


class SexpReader(object):
    def __init__(self, port):
        object.__init__(self)
        self.port = port
        self.nest_level = 0
        self.line = 1
        self.commands = ("'", '(', ',', '`', '"', ';', '@', '|', '[', '{')
        self.white_spaces = (' ', '\r', '\n', '\t')
        self.separations = self.commands + self.white_spaces + (')', '|', ']', '}', EOF, '#')
        self.reader = SexpReader
        self.reader_macro_table = {
            '(': self.make_list_macro,
            '{': self.make_dict_macro,
            '!': self.make_ref_macro,
            'r': self.make_regexp_macro
        }

    def make_list_macro(self):
        sexp = [MAKE_LIST]
        sexp.extend(self.read_delimited_list(')'))
        return sexp

    def make_dict_macro(self):
        sexp = [MAKE_DICT]
        sexp.extend(self.read_delimited_list('}'))
        return sexp

    def make_regexp_macro(self):
        self.port.get_char()
        return RE_COMPILE, self.get_sexp()

    def make_ref_macro(self):
        self.port.get_char()
        return GET_REF, self.get_sexp()

    def has_reader_macro(self, char):
        return char in self.reader_macro_table.keys()

    def get_reader_macro(self, char):
        return self.reader_macro_table[char]

    def get_str(self):
        retval = ''
        while not self.port.next_char() == '"':
            char = self.port.get_char()
            if char is EOF:
                raise Exception('Read error at "' + self.port.filename + '":line ' +
                                str(self.line) + ' : EOF encountered in a string literal')
            if char == '\\':
                try:
                    char = self.port.get_char()
                    if char == 'n':
                        char = "\n"
                    elif char == 't':
                        char = "\t"
                    elif char == 'r':
                        char = "\r"
                except:
                    raise Exception("...")
            retval = retval + char
        self.port.get_char()
        return retval

    def skip_white_spaces(self):
        while self.port.next_char() in self.white_spaces:
            if self.port.next_char() == '\n':
                self.line += 1
            self.port.get_char()

    def skip_until_newline(self):
        while not self.port.next_char() in {'\n', EOF}:
            self.port.get_char()

    def read_until_separation(self):
        retval = ''
        line = self.line
        self.skip_white_spaces()
        while not self.port.next_char() in self.separations:
            retval = retval + self.port.get_char()
        self.skip_white_spaces()
        return retval, line

    def read_delimited_list(self, right_delim):
        self.nest_level += 1
        self.port.get_char()
        retval = []
        while True:
            tkn, _ = self.read_until_separation()
            if tkn != '':
                sexp = self.get_sexp(tkn)
                if sexp is not COMMENT:
                    retval.append(sexp)
            if self.port.next_char() == right_delim:
                self.nest_level -= 1
                self.port.get_char()
                break
            if self.port.next_char() == EOF:
                break
            sexp = self.get_sexp()
            if sexp is not COMMENT:
                retval.append(sexp)
        return retval

    def _compute_underscore_max_num(self, exps):
        max_num = 0
        for exp in exps:
            if isinstance(exp, Symbol) and exp.name.startswith('_'):
                try:
                    n = int(exp.name[1:])
                except:
                    n = 1
            elif issequence_except_str(exp):
                n = self._compute_underscore_max_num(exp)
            else:
                n = 0

            if n > max_num:
                max_num = n
        return max_num

    def _create_underscore_args(self, exps):
        max_num = self._compute_underscore_max_num(exps)
        if max_num == 1:
            return [UNDERSCORE]
        else:
            return [Symbol('_' + str(n)) for n in range(1, max_num + 1)]

    def get_sexp(self, tkn=None):
        if tkn is None:
            tkn, line = self.read_until_separation()
        else:
            line = self.line

        if tkn is EOF:
            return EOF
        elif tkn != '':
            return to_obj(tkn, line)
        elif self.port.next_char() == ';':
            self.skip_until_newline()
            return COMMENT
        elif self.port.next_char() == '(':
            return self.read_delimited_list(')')
        elif self.port.next_char() == '|':
            rest_exps = self.read_delimited_list('|')
            return [FN, self._create_underscore_args(rest_exps), rest_exps]
        elif self.port.next_char() == '[':
            sexp = [MAKE_TUPLE]
            sexp.extend(self.read_delimited_list(']'))
            return sexp
        elif self.port.next_char() == '{':
            sexp = [TABLE]
            sexp.extend(self.read_delimited_list('}'))
            return sexp
        elif self.port.next_char() == "'":
            self.port.get_char()
            sexp = self.get_sexp()
            while sexp is COMMENT:
                sexp = self.get_sexp()
            return [QUOTE, sexp]
        elif self.port.next_char() == '`':
            self.port.get_char()
            sexp = self.get_sexp()
            while sexp is COMMENT:
                sexp = self.get_sexp()
            return [QUASIQUOTE, sexp]
        elif self.port.next_char() == ',':
            self.port.get_char()
            if self.port.next_char() == '@':
                self.port.get_char()
                sexp = self.get_sexp()
                while sexp is COMMENT:
                    sexp = self.get_sexp()
                return [UNQUOTE_SPLICING, sexp]
            else:
                sexp = self.get_sexp()
                while sexp is COMMENT:
                    sexp = self.get_sexp()
                return [UNQUOTE, sexp]
        elif self.port.next_char() == '"':
            self.port.get_char()
            return self.get_str()
        elif self.port.next_char() == '#':
            if self.has_reader_macro(self.port.next_next_char()):
                reader_macro = self.get_reader_macro(self.port.next_next_char())
                self.port.get_char()
                return reader_macro()
            else:
                raise ReadError(self.port.filename, self.line, self.nest_level)
        elif self.port.next_char() == ')':
            raise ReadError(self.port.filename, self.line, self.nest_level)
        elif self.port.next_char() == '@':
            self.port.get_char()
            with_decorator_sexp = [WITH_DECORATOR]
            self.nest_level += 1
            while True:
                decorator_sexp = self.get_sexp()
                if decorator_sexp is EOF:
                    break
                with_decorator_sexp.append(decorator_sexp)
                if issequence_except_str(decorator_sexp) and decorator_sexp[0] == DEF:
                    self.nest_level -= 1
                    break
            return with_decorator_sexp
        else:
            return EOF


def emit_sexp(sexpr):
    ol = []
    stack = [sexpr]

    while len(stack) > 0:
        sexpr = stack.pop()
        if is_tuple_or_list(sexpr):
            stack.append(RPARA)
            rsexpr = []
            for sub in sexpr:
                rsexpr.insert(0, sub)
            stack.extend(rsexpr)
            stack.append(LPARA)
        else:
            ol.append(sexpr)

    retval = ''
    oldsitem = ''
    for item in ol:
        sitem = repr(item)
        if sitem[0] == "'" and sitem[-1] == "'":
            sitem = sitem.replace('"', "\\\"")
            sitem = '"' + sitem[1:-1] + '"'
        if not ((sitem == ')') or (oldsitem == '(')):
            oldsitem = sitem
            sitem = ' ' + sitem
        else:
            oldsitem = sitem
        retval += sitem
    return retval[1:]


#---------------------------------------------------------
# Error
class UnquoteSplicingError(Exception):
    def __init__(self):
        self.msg = 'unquote-splicing appeared in invalid context'

    def __repr__(self):
        return self.msg

    def __str__(self):
        return self.msg


class MochiSyntaxError(Exception):
    def __init__(self, exp, filename):
        lineno = 0
        if hasattr(exp, 'lineno'):
            lineno = exp.lineno
        elif issequence_except_str(exp) and hasattr(exp[0], 'lineno'):
            lineno = exp[0].lineno
        self.msg = 'syntax error on ' + \
                   'file "' + filename + '", ' + 'line ' + str(lineno) + ': ' + emit_sexp(exp)

    def __str__(self):
        return self.msg

    def __repr__(self):
        return self.msg


class DuplicatedDefError(Exception):
    def __init__(self, exp, filename):
        lineno = 0
        if hasattr(exp, 'lineno'):
            lineno = exp.lineno
        elif issequence_except_str(exp) and hasattr(exp[0], 'lineno'):
            lineno = exp[0].lineno
        self.msg = 'duplicated-def error: ' + \
                   'file "' + filename + '", ' + 'line ' + str(lineno) + ': ' + emit_sexp(exp)

    def __str__(self):
        return self.msg

    def __repr__(self):
        return self.msg


#---------------------------------------------------------
# Port Classes
class InputPort(object):
    def __init__(self, file=sys.__stdin__):
        self._closed = False
        self.file = file
        self.filename = '<string>' if isinstance(file, StringIO) else file.name
        self.buffer = ''

    def get_char(self):
        if len(self.buffer) == 0:
            c = self.file.read(1)
            if len(c) == 0:
                return EOF
            return c
        retval = self.buffer[0]
        self.buffer = self.buffer[1:]
        return retval

    def next_char(self):
        if len(self.buffer) == 0:
            c = self.file.read(1)
            if len(c) == 0:
                return EOF
            self.buffer = c
        return self.buffer[0]

    def next_next_char(self):
        if len(self.buffer) == 0:
            c = self.file.read(1)
            if len(c) == 0:
                return EOF
            self.buffer = c
            c = self.file.read(1)
            if len(c) == 0:
                return EOF
            self.buffer = self.buffer + c
        elif len(self.buffer) == 1:
            c = self.file.read(1)
            if len(c) == 0:
                return EOF
            self.buffer = self.buffer + c
        return self.buffer[1]

    def _read(self):
        if self._closed:
            raise Exception('error!')
        else:
            sr = SexpReader(self)
            while True:
                retval = sr.get_sexp()
                if retval is not COMMENT:
                    break
            return retval

    def _close(self):
        self._closed = True
        return None


class OutputPort(object):
    def __init__(self, file=sys.__stdout__):
        self.file = file

    def display(self, arg):
        port = self.file
        sexpr = arg
        ol = []
        stack = [sexpr]

        while len(stack) > 0:
            sexpr = stack.pop()
            if is_tuple_or_list(sexpr):
                stack.append(RPARA)
                rsexpr = []
                for sub in sexpr:
                    rsexpr.insert(0, sub)
                stack.extend(rsexpr)
                stack.append(LPARA)
            else:
                ol.append(sexpr)

        retval = ''
        olditem = ''
        for item in ol:
            if type(item) is str:
                if not ((item is RPARA) or (olditem is LPARA)):
                    item = ' ' + item
                retval += item
            else:
                sitem = str(item)
                if not ((item is RPARA) or (olditem is LPARA)):
                    sitem = ' ' + sitem
                retval += sitem
            olditem = item
        port.write(retval[1:])
        return None

    def write(self, obj):
        self.file.write(str(obj))  # emit_sexp(obj))
        return None


current_input_port = InputPort()
current_output_port = OutputPort()
py_print = print


@builtin
def display(obj):
    return current_output_port.display(obj)


@builtin
def write(obj):
    return current_output_port.write(obj)


@builtin
def format(s, *args, **kwargs):
    return s.format(*args, **kwargs)


@builtin
def zero(arg):
    return arg == 0


@builtin
def positive(arg):
    return arg > 0


@builtin
def negative(arg):
    return arg < 0


@builtin
def odd(arg):
    return type(arg) is int and bool(arg % 2)


@builtin
def even(arg):
    return type(arg) is int and not bool(arg % 2)


@builtin
def caar(seq):
    return seq[0][0]


@builtin
@builtin_rename('second')
def cadr(seq):
    return seq[1]


@builtin
def cdar(seq):
    return seq[0][1:]


@builtin
def cddr(seq):
    return seq[2:]


@builtin
def caaar(seq):
    return seq[0][0][0]


@builtin
def caadr(seq):
    return seq[1][0]


@builtin
def cadar(seq):
    return seq[0][1]


@builtin
@builtin_rename('third')
def caddr(seq):
    return seq[2]


@builtin
@builtin_rename('fourth')
def cadddr(seq):
    return seq[3]


@builtin
def cdaar(seq):
    return seq[0][0][1:]


@builtin
def cdadr(seq):
    return seq[1][1:]


@builtin
def cddar(seq):
    return seq[0][2:]


@builtin
def cdddr(seq):
    return seq[3:]


@builtin
def null(obj):
    return obj == () or obj is None or (issequence(obj) and len(obj) == 0)


@builtin
class cell(Sequence):
    __slots__ = ('_head', '_tail')

    def __init__(self, head, tail=None):
        self._head = head
        if isinstance(tail, cell) or (tail is None):
            self._tail = tail
        else:
            raise TypeError('tail: expected None or cell, but ' + type(tail) + ' found')

    def __getitem__(self, index):
        if isinstance(index, slice):
            start = index.start
            pos_index = start if start >= 0 else self.__len__() + start
        else:
            start = None
            pos_index = index if index >= 0 else self.__len__() + index

        item = self
        while pos_index > 0:
            if isinstance(item, cell):
                item = item._tail
            else:
                raise IndexError('index out of range')
            pos_index -= 1

        if start is None:
            if item is None:
                raise IndexError('index out of range')
            return item._head

        return item

    def __len__(self):
        length = 0
        item = self
        while True:
            if isinstance(item, cell):
                item = item._tail
                length += 1
            else:
                return length

    def __repr__(self):
        repr_strs = ['L(']
        for item in self:
            repr_strs.append(repr(item))
            repr_strs.append(' ')
        if len(repr_strs) == 0:
            repr_strs.append(')')
        else:
            repr_strs[-1] = ')'

        return ''.join(repr_strs)


@builtin_rename('clist')
@builtin_rename('cell*')
def make_cell_list(*args):
    lis = None
    for arg in reversed(args):
        lis = cell(arg, lis)
    return lis


@builtin
class lazyseq(Sequence):
    def __init__(self, iterable):
        self._cache = []
        self._iterator = iter(iterable)

    def __getitem__(self, index):
        if isinstance(index, slice):
            start = index.start if index.start >= 0 else len(self) + index.start
            stop = len(self) if index.stop is None else index.stop if index.stop >= 0 else len(self) + index.stop
            try:
                len_cache = len(self._cache)
                while len_cache < stop:
                    self._cache.append(next(self._iterator))
                    len_cache += 1
            except StopIteration:
                raise IndexError('index out of range')
            return self._cache[slice(start, stop, index.step)]
        else:
            pos_index = index if index >= 0 else len(self) + index
            try:
                len_cache = len(self._cache)
                while len_cache <= pos_index:
                    self._cache.append(next(self._iterator))
                    len_cache += 1
            except StopIteration:
                raise IndexError('index out of range')
            return self._cache[pos_index]

    def __len__(self):
        self._cache.extend(list(self._iterator))
        return len(self._cache)

    def __repr__(self):
        return 'lazyseq(' + repr(self._iterator) + ')'

    def __call__(self, index, *rest):
        len_rest = len(rest)
        if len_rest == 1:
            return self.__getitem__(slice(index, rest[0]))
        elif len_rest == 2:
            return self.__getitem__(slice(index, rest[0], rest[1]))
        elif len_rest >= 3:
            slice(index, *rest)
        else:
            return self.__getitem__(index)


@builtin
def table(*keyvalues, **somedict):
    return pmap(dict(chunks(keyvalues, 2), **somedict))


@builtin
def listtab(lis):
    return table(*lis)


@builtin
def tablist(tabl):
    return tuple(tabl.items())


@builtin
def getslice(seq, start, end=None, step=1):
    return seq[start:end:step]


@builtin
def curried(func):
    arg_count = func.__code__.co_argcount
    l = ['lambda arg_{0}: '.format(n) for n in range(arg_count)]
    l.append('func(')
    l.extend(['arg_{0}, '.format(n) for n in range(arg_count - 1)])
    l.extend(['arg_', str(arg_count - 1), ')'])
    return py_eval(''.join(l), {'func': func})


@builtin
def rcurried(func):
    argcount = func.__code__.co_argcount
    l = ['lambda arg_{0}: '.format(n) for n in range(argcount - 1, -1, -1)]
    l.append('func(')
    l.extend(['arg_{0}, '.format(n) for n in range(argcount - 1)])
    l.extend(['arg_', str(argcount - 1), ')'])
    return py_eval(''.join(l), {'func': func})


@builtin_rename('auto-partial')
def auto_partial(func):
    if isinstance(func, partial):
        args_count = 0
        inner_func = func
        while isinstance(inner_func, partial):
            args_count += len(inner_func.args)
            inner_func = inner_func.func
        original_func_argcount = inner_func.__code__.co_argcount
        func_argcount = original_func_argcount - args_count
    else:
        func_argcount = func.__code__.co_argcount

    def _partial(*args):
        if (func_argcount - len(args)) <= 0:
            return func(*args)
        else:
            return auto_partial(partial(func, *args))

    return _partial


@builtin
def apply(func, args, kwargs=None):
    if kwargs is None:
        return func(*args)
    else:
        return func(*args, **kwargs)


@builtin
def compose(*funcs, unpack=False):
    reversed_funcs = list(reversed(funcs))

    def composed_func(*args, **kwargs):
        retval = reversed_funcs[0](*args, **kwargs)
        if unpack:
            for func in reversed_funcs[1:]:
                retval = func(*retval)
                return retval
        else:
            for func in reversed_funcs[1:]:
                retval = func(retval)
                return retval

    return composed_func


@builtin
def conjoin(*funcs):
    def conjoined_func(*args, **kwargs):
        retval = True
        for func in funcs:
            retval = retval and func(*args, **kwargs)
            if not retval:
                break
        return retval

    return conjoined_func


@builtin
def disjoin(*funcs):
    def disjoined_func(*args, **kwargs):
        retval = False
        for func in funcs:
            retval = retval or func(*args, **kwargs)
            if retval:
                break
        return retval

    return disjoined_func


@builtin
def compare(comparer, scorer):
    return lambda arg1, arg2: comparer(scorer(arg1), scorer(arg2))


@builtin
def complement(func):
    return lambda *args, **kwargs: not (func(*args, **kwargs))


@builtin
def ref(value):
    # getter
    def _0():
        return value

    # setter
    def _1(update_func):
        nonlocal value
        value = update_func(value)
        return value

    return _0, _1


#@builtin_rename('set!')
#def update_ref(ref, update_func):
#    return ref[1](update_func)


struct_ids = []


@builtin_rename('define-struct')
def define_struct(typename, field_names, verbose=False, rename=False):
    struct_ids.append(typename)
    # return namedtuple(typename, field_names, verbose, rename)
    return pclass(field_names, typename, verbose)


@builtin_rename('struct-id?')
def is_struct_id(name):
    return name in struct_ids


record_ids = ['Record']


@builtin_rename('record-id?')
def is_record(name):
    return name in record_ids


#---------------------------------------------------------
VERSION = '0.0.1'

syntax_table = {}


def syntax(*names):
    def _syntax(func):
        for name in names:
            syntax_table[name] = func
            if name not in global_env.keys():
                global_env[name] = 'syntax-' + name
        return func

    return _syntax


op_ast_map = {'+': ast.Add(),
              '-': ast.Sub(),
              '*': ast.Mult(),
              '/': ast.Div(),
              '%': ast.Mod(),
              '**': ast.Pow(),
              '<<': ast.LShift(),
              '>>': ast.RShift(),
              'bitor': ast.BitOr(),
              '^': ast.BitXor(),
              '&': ast.BitAnd(),
              '//': ast.FloorDiv(),
              '==': ast.Eq(),
              '!=': ast.NotEq(),
              '<': ast.Lt(),
              '<=': ast.LtE(),
              '>': ast.Gt(),
              '>=': ast.GtE(),
              'is': ast.Is(),
              'is-not': ast.IsNot(),
              'in': ast.In(),
              'not-in': ast.NotIn(),
              'and': ast.And(),
              'or': ast.Or()}


def add_binding_name(binding_name, file_name):
    return binding_name_set_stack[-1].add_binding_name(binding_name, file_name)


def _check_duplicated_binding_name(symbol, filename, lis):
    constant_name = symbol.name
    for constant_name_set in reversed(lis):
        if constant_name in constant_name_set:
            raise DuplicatedDefError(symbol, filename)
        if not constant_name_set.quasi:
            break
    lis[-1].add(constant_name)
    return symbol


def check_duplicated_binding_name(symbol, filename):
    return symbol
    # return _check_duplicated_binding_name(symbol, filename, binding_name_set_stack)


def check_duplicated_binding_name_outer(symbol, filename):
    binding_name_set_stack[-2].add(symbol.name)
    return symbol


def check_duplicated_binding(sexps, filename):
    for sexp in sexps:
        if not issequence_except_str(sexp):
            continue
        if not isinstance(sexp[0], Symbol):
            continue
        tag = sexp[0].name
        current_set = binding_name_set_stack[-1]
        if tag in {'_val', 'def', 'mac'}:
            binding_symbol = sexp[1]
            binding_name = binding_symbol.name
            if binding_name in current_set:
                raise DuplicatedDefError(binding_symbol, filename)
            current_set.add(binding_name)
    return sexps


class Scope(dict):
    def __init__(self, quasi=False):
        self.quasi = quasi
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def merge_children(self):
        merge_map = {}
        for child in self.children:
            merge_map.update(child)
        for binding_name in merge_map.keys():
            self.add_binding_name(binding_name, merge_map[binding_name])

    def add_binding_name(self, binding_name, file_name):
        # TODO _gs
        if not binding_name.startswith('_gs') and binding_name in self:
            raise DuplicatedDefError(binding_name, file_name)
        self[binding_name] = file_name
        return binding_name


def lexical_scope(quasi=False):
    def _lexical_scope(func):
        def __lexical_scope(*args, **kwargs):
            try:
                scope = Scope(quasi)
                if quasi:
                    binding_name_set_stack[-1].add_child(scope)
                binding_name_set_stack.append(scope)
                return func(*args, **kwargs)
            finally:
                if binding_name_set_stack[-1] is scope:
                    binding_name_set_stack.pop()

        return __lexical_scope

    return _lexical_scope


def flatten_list(lis):
    i = 0
    while i < len(lis):
        while isinstance(lis[i], Iterable):
            if not lis[i]:
                lis.pop(i)
                i -= 1
                break
            else:
                lis[i:i + 1] = lis[i]
        i += 1
    return lis


@builtin
def unquote_splice(lis):
    newlis = []
    splicing_flag = False
    for item in lis:
        if type(item) is Symbol and item.name == '_mochi_unquote-splicing':
            splicing_flag = True
        else:
            if splicing_flag:
                newlis.extend(item)
                splicing_flag = False
            else:
                newlis.append(item)

    if type(lis) is list:
        return newlis
    elif type(lis) is tuple:
        return tuple(newlis)
    else:
        try:
            return type(lis)(newlis)
        except:
            raise UnquoteSplicingError()


@builtin
def tuple_it(obj):
    return tuple(map(tuple_it, obj)) if isinstance(obj, MutableSequence) else obj


if IS_PYTHON_34:
    expr_and_stmt_ast = (ast.Expr, ast.If, ast.For, ast.FunctionDef, ast.Assign, ast.Delete, ast.Try, ast.Raise,
                         ast.With, ast.While, ast.Break, ast.Return, ast.Continue, ast.ClassDef,
                         ast.Import, ast.ImportFrom, ast.Pass)
else:
    expr_and_stmt_ast = (ast.Expr, ast.If, ast.For, ast.FunctionDef, ast.Assign, ast.Delete, ast.TryFinally,
                         ast.TryExcept, ast.Raise, ast.With, ast.While, ast.Break, ast.Return, ast.Continue,
                         ast.ClassDef, ast.Import, ast.ImportFrom, ast.Pass)

required_filename_set = set()


class Translator(object):
    def __init__(self, macro_table={}, filename='<string>'):
        self.hidden_vars = []
        self.macro_table = macro_table
        self.filename = filename

    def translate_file(self, filename):
        body = []
        self.filename = filename
        with open(filename, 'r') as f:
            sexps = parse(lex(f.read()))
            chdir(normpath(str(Path(filename).parent)))
            for sexp in sexps:
                if isinstance(sexp, MutableSequence):
                    sexp = tuple_it(sexp)
                if sexp is COMMENT:
                    continue
                pre, value = self.translate(sexp)
                body.extend([self.enclose(exp, True) for exp in pre])
                body.append(self.enclose(value, True))
        return ast.Module(body=body)

    def translate_loaded_file(self, filename):
        body = []
        self.filename = filename
        with open(filename, 'r') as f:
            sexps = parse(lex(f.read()))
            for sexp in sexps:
                if isinstance(sexp, MutableSequence):
                    sexp = tuple_it(sexp)
                if sexp is COMMENT:
                    continue
                pre, value = self.translate(sexp)
                body.extend(pre)
                body.append(value)
        return body

    def translate_required_file(self, filename):
        if filename in required_filename_set:
            return self.translate_ref(NONE_SYM)[1],
        else:
            required_filename_set.add(filename)
            return self.translate_loaded_file(filename)

    #    def translate_loaded_files(self, file_symbols):
    def translate_loaded_files(self, file_paths):
        body = []
        for file_path in file_paths:
            if not isinstance(file_path, str):
                raise MochiSyntaxError(file_path, self.filename)
            body.extend(self.translate_loaded_file(abspath(file_path)))
        return body, self.translate_ref(NONE_SYM)[1]

    def translate_required_files(self, file_paths):
        body = []
        for file_path in file_paths:
            if not isinstance(file_path, str):
                raise MochiSyntaxError(file_path, self.filename)
            body.extend(self.translate_required_file(abspath(file_path)))
        return body, self.translate_ref(NONE_SYM)[1]

    def translate_sexp_to_interact(self, sexp):
        pre, value = self.translate(sexp, False)
        body = [self.enclose(exp, True) for exp in pre]
        write_value = ast.Call(func=ast.Name(id='write',
                                             ctx=ast.Load(),
                                             lineno=1,
                                             col_offset=0),
                               ctx=ast.Load(),
                               args=[value],
                               keywords=[],
                               starargs=None,
                               kwargs=None,
                               lineno=1,
                               col_offset=0)
        body.append(self.enclose(write_value, True))
        return ast.Module(body=body)

    def translate_sexp(self, sexp):
        # TODO value
        pre, value = self.translate(sexp, False)
        body = [self.enclose(exp, True) for exp in pre]
        return ast.Module(body=body)

    def is_self_evaluating(self, exp):
        return isinstance(exp, (int, float, complex, str, bool, type(None), Keyword))

    def translate_self_evaluating(self, exp):
        if isinstance(exp, (int, float, complex)):
            if hasattr(exp, 'value'):
                return EMPTY, ast.Num(exp.value,
                                      lineno=exp.lineno,
                                      col_offset=0)
            else:
                return EMPTY, ast.Num(exp,
                                      lineno=0,
                                      col_offset=0)

        expType = type(exp)

        if expType is str:
            return EMPTY, ast.Str(exp,
                                  lineno=0,
                                  col_offset=0)

        if expType is Keyword:
            #raise SyntaxError('keyword!', self.filename)
            modast = ast.parse(
                'Keyword("{name}", {lineno})'.format(name=exp.name, lineno=exp.lineno))
            return EMPTY, modast.body[0].value

    def _translate_items(self, items):
        return flatten_list([self.translate(item, False) for item in items])

    def is_variable(self, exp):
        return type(exp) is Symbol

    def is_get_attrs(self, exp):
        return type(exp) is Symbol and len(exp.name.split('.')) > 1

    def translate_ref(self, exp):
        return EMPTY, ast.Name(id=exp.name,
                               ctx=ast.Load(),
                               lineno=exp.lineno,
                               col_offset=0)

    def translate_atom(self, exp):
        if self.is_self_evaluating(exp):
            return self.translate_self_evaluating(exp)
        elif self.is_get_attrs(exp):
            parts = exp.name.split('.')
            parts[0] = Symbol(parts[0])
            return self.translate(
                reduce(lambda a, b: (Symbol('getattr'), a, b),
                       parts), False)
        elif self.is_variable(exp):
            return self.translate_ref(exp)

    @syntax('_val')
    def translate_def(self, exp):
        if len(exp) != 3:
            raise MochiSyntaxError(exp, self.filename)
        return self.translate_assign(exp)

    def create_assign_target(self, symbol):
        return ast.Name(id=symbol.name,
                        ctx=ast.Store(),
                        lineno=symbol.lineno,
                        col_offset=0)

    def create_assign_targets(self, seq):
        elts = []
        for index, item in enumerate(seq):
            if isinstance(item, Symbol):
                # TODO errorcheck index + 1 != lastIndex
                if item == VARG:
                    elts.append(
                        ast.Starred(
                            value=self.create_assign_target(
                                seq[index + 1]),
                            ctx=ast.Store(),
                            lineno=0,  # TODO
                            col_offset=0))
                    break
                elts.append(self.create_assign_target(item))
            elif issequence_except_str(item):
                if len(item) == 0:
                    raise MochiSyntaxError("can't bind to ()", self.filename)
                #if item[0] == 'getattr':
                #    attr_exp = self.translate_getattr(item)[1]
                #    attr_exp.ctx = ast.Store()
                #    elts.append(attr_exp)
                #else:
                elts.append(self.create_assign_targets(item))
            else:
                raise MochiSyntaxError(item, self.filename)
        return ast.Tuple(elts=elts,
                         ctx=ast.Store(),
                         lineno=0,  # TODO
                         col_offset=0)

    #@syntax('set')
    def translate_assign(self, exp, visible=True):
        if len(exp) != 3:
            raise MochiSyntaxError(exp, self.filename)

        left = exp[1]
        left_type = type(left)
        if left_type is Symbol:
            targets = [self.create_assign_target(left)]
            ref_symbol = left
            if not visible:
                self.hidden_vars.append(ref_symbol.name)
        elif issequence_except_str(left):
            targets = [self.create_assign_targets(left)]
            ref_symbol = NONE_SYM
        else:
            raise MochiSyntaxError(exp, self.filename)

        pre = []
        right_value_builder, right_value = self.translate(exp[2], False)
        if type(right_value) is ast.Expr:
            right_value = right_value.value
        assign = ast.Assign(targets=targets,
                            value=right_value,
                            lineno=right_value.lineno,
                            col_offset=0)
        pre.extend(right_value_builder)
        pre.append(assign)
        _, ref = self.translate_ref(ref_symbol)
        return pre, ref

    #@syntax('del')
    #def translate_del(self, exp):
    #    return (ast.Delete(targets=[ast.Name(id=exp[1].name,
    #                                         lineno=exp[1].lineno,
    #                                         col_offset=exp[1].col_offset,
    #                                         ctx=ast.Del())],
    #                       lineno=exp[0].lineno,
    #                       col_offset=exp[0].col_offset),), self.translate_ref(NONE_SYM)[1]

    @syntax('yield')
    def translate_yield(self, exp):
        if len(exp) != 2:
            raise MochiSyntaxError(exp, self.filename)

        pre, value = self.translate(exp[1], False)
        if type(value) is ast.Expr:
            value = value.value
        yield_node = ast.Yield(value=value,
                               lineno=exp[0].lineno,
                               col_offset=0)
        return pre, yield_node

    @syntax('yield-from')
    def translate_yield_from(self, exp):
        if len(exp) != 2:
            raise MochiSyntaxError(exp, self.filename)

        pre, value = self.translate(exp[1], False)
        if type(value) is ast.Expr:
            value = value.value
        yield_from_node = ast.YieldFrom(value=value,
                                        lineno=exp[0].lineno,
                                        col_offset=0)
        return pre, yield_from_node

    def _translate_args(self, args):
        return [ast.arg(arg=arg.name,
                        annotation=None,
                        lineno=arg.lineno,
                        col_offset=0) for arg in args]

    def _translate_sequence(self, exps, enclose=True):
        seq = []
        for exp in exps:
            pre, value = self.translate(exp, enclose)
            seq.extend(pre)
            seq.append(value)
        return seq

    @syntax('do')
    def translate_do(self, exp, enclose=False):
        seq = self._translate_sequence(exp[1:], enclose)
        return seq[:-1], seq[-1]

    @syntax('if')
    def translate_if(self, exp):
        if not len(exp) >= 3:
            raise MochiSyntaxError(exp, self.filename)
        temp_var_symbol = get_temp_name()
        return self._translate_if(exp, temp_var_symbol)

    def _translate_if(self, exp, temp_var_symbol):
        if_ast = None
        cur_if_ast = None
        pre = []
        for i, v in enumerate(exp):
            if i == 0:
                pass
            elif (i % 2) == 0:
                pre_test, test = self.translate(exp[i - 1], False)
                pre.extend(pre_test)
                if if_ast is None:
                    if issequence_except_str(v) and v[0] == Symbol('return'):
                        body, ref = self.translate_return(v)
                        if_ast = ast.If(test=test,
                                        body=body,
                                        orelse=[],
                                        lineno=exp[0].lineno,
                                        col_offset=0)
                    else:
                        body, ref = self.translate_assign(('dummy', temp_var_symbol, v), visible=False)
                        if_ast = ast.If(test=test,
                                        body=body,
                                        orelse=[],
                                        lineno=exp[0].lineno,
                                        col_offset=0)
                    cur_if_ast = if_ast
                else:
                    else_if = [Symbol('if'), exp[i - 1], v]
                    else_body, ref = self._translate_if(else_if, temp_var_symbol)
                    cur_if_ast.orelse = else_body
                    cur_if_ast = else_body[0]
            elif i == (len(exp) - 1):
                if issequence_except_str(v) and v[0] == Symbol('return'):
                    else_body, ref = self.translate_return(v)
                    cur_if_ast.orelse = else_body
                else:
                    else_body, ref = self.translate_assign(
                        ('dummy', temp_var_symbol, v), visible=False)
                    cur_if_ast.orelse = else_body

        if len(if_ast.orelse) == 0:
            else_body, ref = self.translate_assign(
                ('dummy', temp_var_symbol, EMPTY_SYM), visible=False)
            if_ast.orelse = else_body

        pre.append(if_ast)
        return pre, ref

    @syntax('+', '-', '*', '/', '%', '**',
            '<<', '>>', 'bitor', '^', '&', '//')
    def translate_bin_op(self, exp):
        if not len(exp) >= 2:
            raise MochiSyntaxError(exp, self.filename)

        if len(exp) == 2:
            return self.translate(exp[1], False)

        op_symbol = exp[0]
        op_name = op_symbol.name
        op = op_ast_map[op_name]

        if len(exp) == 3:
            left, right = exp[1], exp[2]
            left_pre, left_value = self.translate(left, False)
            right_pre, right_value = self.translate(right, False)
            pre = left_pre + right_pre
            return pre, ast.BinOp(op=op,
                                  left=left_value,
                                  right=right_value,
                                  lineno=op_symbol.lineno,
                                  col_offset=0)
        else:
            rest, right = exp[0:-1], exp[-1]
            rest_pre, rest_value = self.translate(rest, False)
            right_pre, right_value = self.translate(right, False)
            pre = rest_pre + right_pre
            return pre, ast.BinOp(op=op,
                                  left=rest_value,
                                  right=right_value,
                                  lineno=op_symbol.lineno,
                                  col_offset=0)

    @syntax('and', 'or')
    def translate_bool_op(self, exp):
        if not len(exp) >= 3:
            raise MochiSyntaxError(exp, self.filename)

        op_symbol = exp[0]
        op_name = op_symbol.name
        op = op_ast_map[op_name]

        pre = []
        values = []
        for value in exp[1:]:
            pre_value, value_value = self.translate(value, False)
            pre += pre_value
            values.append(value_value)

        return pre, ast.BoolOp(op=op,
                               values=values,
                               lineno=op_symbol.lineno,
                               col_offset=0)

    @syntax('=', '!=', '<', '<=', '>', '>=',
            'is', 'is-not', 'in', 'not-in')
    def translate_compare(self, exp):
        if len(exp) < 3:
            raise MochiSyntaxError(exp, self.filename)

        op_symbol = exp[0]
        op_name = op_symbol.name

        left, rights = exp[1], exp[2:]
        ops = [op_ast_map[op_name]] * len(rights)
        pre, left_value = self.translate(left, False)
        right_values = []
        for right in rights:
            right_pre, right_value = self.translate(right, False)
            pre = pre + right_pre
            right_values.append(right_value)

        return pre, ast.Compare(ops=ops,
                                left=left_value,
                                comparators=right_values,
                                lineno=op_symbol.lineno,
                                col_offset=0)

    def make_return(self, exp):
        pre, value = self.translate(exp, False)
        if type(value) is ast.Expr:
            value = value.value
        ret = ast.Return(value=value,
                         lineno=0,  # exp.lineno,
                         col_offset=0)
        return pre, ret

    @syntax('for', 'each')
    def translate_foreach(self, exp):
        if not (len(exp) >= 4):
            raise MochiSyntaxError(exp, self.filename)

        target_exp = exp[1]
        if isinstance(target_exp, Symbol):
            target = self.create_assign_target(target_exp)
        elif issequence_except_str(target_exp):
            target = self.create_assign_targets(target_exp)
        else:
            raise MochiSyntaxError(exp, self.filename)

        pre = []
        iter_pre, iter_value = self.translate(exp[2], False)
        pre.extend(iter_pre)

        body = self._translate_sequence(exp[3:])
        pre.append(ast.For(target=target,
                           iter=iter_value,
                           body=body,
                           orelse=[],
                           lineno=exp[0].lineno,
                           col_offset=0))

        _, ref = self.translate_ref(NONE_SYM)
        return pre, ref

    @syntax('while')
    def translate_while(self, exp):
        if not (len(exp) >= 3):
            raise SyntaxError(exp, self.filename)

        test_exp = exp[1]
        body_exps = exp[2:]

        pre = []

        test_pre, test_value = self.translate(test_exp, False)
        pre.extend(test_pre)

        body = self._translate_sequence(body_exps)
        pre.append(ast.While(test=test_value,
                             body=body,
                             orelse=[],
                             lineno=exp[0].lineno,
                             col_offset=0))

        _, ref = self.translate_ref(NONE_SYM)
        return pre, ref

    @syntax('break')
    def translate_break(self, exp):
        if len(exp) > 1:
            raise MochiSyntaxError(exp, self.filename)

        return (), ast.Break(lineno=exp[0].lineno,
                             col_offset=0)

    @syntax('continue')
    def translate_continue(self, exp):
        if len(exp) > 1:
            raise MochiSyntaxError(exp, self.filename)

        return (), ast.Continue(lineno=exp[0].lineno,
                                col_offset=0)

    def _translate_get_index(self, exp):
        pre = []
        target_pre, target_value = self.translate(exp[1], False)
        pre.extend(target_pre)
        index_pre, index_value = self.translate(exp[2], False)
        pre.extend(index_pre)
        return pre, ast.Subscript(value=target_value,
                                  slice=ast.Index(value=index_value),
                                  ctx=ast.Load(),
                                  lineno=exp[0].lineno,
                                  col_offset=0)

    def _translate_get_slice(self, lineno, target, start=None, stop=None, step=None):
        pre = []
        target_pre, target_value = self.translate(target, False)
        pre.extend(target_pre)
        start_pre, start_value = self.translate(start, False) if start is not None else ((), None)
        pre.extend(start_pre)
        stop_pre, stop_value = self.translate(stop, False) if stop is not None else ((), None)
        pre.extend(stop_pre)
        step_pre, step_value = self.translate(step, False) if step is not None else ((), None)
        pre.extend(step_pre)
        return pre, ast.Subscript(value=target_value,
                                  slice=ast.Slice(lower=start_value,
                                                  upper=stop_value,
                                                  step=step_value),
                                  ctx=ast.Load(),
                                  lineno=lineno,
                                  col_offset=0)

    @syntax('get')
    def translate_get(self, exp):
        exp_length = len(exp)
        if exp_length == 3:
            return self._translate_get_index(exp)
        elif exp_length == 4:
            return self._translate_get_slice(exp[0].lineno, exp[1], exp[2], exp[3])
        elif exp_length == 5:
            return self._translate_get_slice(exp[0].lineno, exp[1], exp[2], exp[3], exp[4])
        else:
            raise MochiSyntaxError(exp, self.filename)

    @syntax('car', 'first')
    def translate_car(self, exp):
        if len(exp) != 2:
            raise MochiSyntaxError(exp, self.filename)
        return self._translate_get_index((GET, exp[1], 0))

    @syntax('cdr', 'rest')
    def translate_cdr(self, exp):
        if len(exp) != 2:
            raise MochiSyntaxError(exp, self.filename)
        return self._translate_get_slice(exp[0].lineno, exp[1], 1)

    @syntax('last')
    def translate_last(self, exp):
        if len(exp) != 2:
            raise MochiSyntaxError(exp, self.filename)
        return self._translate_get_index((GET, exp[1], -1))

    @syntax('cadr')
    def translate_cadr(self, exp):
        if len(exp) != 2:
            raise MochiSyntaxError(exp, self.filename)
        return self._translate_get_index((GET, exp[1], 1))

    @syntax('getattr')
    def translate_getattr(self, exp):
        return (), ast.Attribute(value=self.translate(exp[1], False)[1],
                                 attr=exp[2],
                                 lineno=exp[0].lineno,
                                 col_offset=exp[0].col_offset,
                                 ctx=ast.Load())

    @syntax('with-decorator')
    def translate_with_decorator(self, exp):
        if not (len(exp) >= 3):
            raise MochiSyntaxError(exp, self.filename)
        return self.translate_defun(exp[-1], decorator_exps=exp[1:-1])

    class DuplicatedBindingChecker(ast.NodeTransformer):
        def __init__(self, filename):
            self.filename = filename

        @lexical_scope()
        def visit_ClassDef(self, node):
            add_binding_name(node.name, self.filename)
            return self.generic_visit(node)

        @lexical_scope(quasi=True)
        def visit_If(self, node):
            self.visit(node.test)
            for body_item in node.body:
                self.visit(body_item)
            binding_name_set_stack.pop()
            for orelse_item in node.orelse:
                if not isinstance(orelse_item, ast.If):
                    new_scope = Scope(quasi=True)
                    binding_name_set_stack[-1].add_child(new_scope)
                    binding_name_set_stack.append(new_scope)
                    self.visit(orelse_item)
                    binding_name_set_stack.pop()
                else:
                    self.visit(orelse_item)
            return node

        def create_init_assign(self, node):
            targets = []
            arg_names = [arg if isinstance(arg, str) else arg.arg for arg in node.args.args]
            if node.args.vararg is not None:
                arg_names.append(node.args.vararg if isinstance(node.args.vararg, str) else node.args.vararg.arg)
            if node.args.kwarg is not None:
                arg_names.append(node.args.kwarg if isinstance(node.args.kwarg, str) else node.args.kwarg.arg)
            for binding_name in node.binding_name_set:
                if binding_name == node.name:
                    continue
                if binding_name in arg_names:
                    continue
                targets.append(ast.Name(id=binding_name,
                                        ctx=ast.Store(),
                                        lineno=0,
                                        col_offset=0))
            if len(targets) > 0:
                if hasattr(ast, 'NameConstant'):
                    right_value = ast.NameConstant(value=None)
                else:
                    right_value = ast.Name(id=NONE_SYM.name,
                                           ctx=ast.Load(),
                                           lineno=0,
                                           col_offset=0)
                return [ast.Assign(targets=targets,
                                   value=right_value,
                                   lineno=0,
                                   col_offset=0)]
            else:
                return []

        @lexical_scope()
        def visit_FunctionDef(self, node):
            if hasattr(node, 'is_visited') and node.is_visited:
                return node
            add_binding_name(node.name, self.filename)
            for arg in node.args.args:
                add_binding_name(arg if isinstance(arg, str) else arg.arg, self.filename)
            if node.args.vararg is not None:
                add_binding_name(node.args.vararg if isinstance(node.args.vararg, str) else node.args.vararg.arg,
                                  self.filename)
            if node.args.kwarg is not None:
                add_binding_name(node.args.kwarg if isinstance(node.args.kwarg) else node.args.kwarg.arg,
                                  self.filename)
            for stmt in node.body:
                self.visit(stmt)
            binding_name_set_stack[-1].merge_children()
            node.binding_name_set = binding_name_set_stack[-1]
            node.body = self.create_init_assign(node) + node.body
            node.is_visited = True
            return node

        def visit_Assign(self, node):
            if not (hasattr(node, 're_assign') and node.re_assign):
                for target in node.targets:
                    add_binding_name(target.id, self.filename)
            return node

        def visit_ExceptHandler(self, node):
            if hasattr(node, 'name'):
                add_binding_name(node.name, self.filename)
            return node

    def _check_duplicated_binding(self, func_ast):
        checker = self.DuplicatedBindingChecker(self.filename)
        return checker.visit(func_ast)

    class SelfTailRecursiveCallTransformer(ast.NodeTransformer):

        optimized = False

        def __init__(self, target_func):
            self.target_func = target_func

        def visit_FunctionDef(self, node):
            if node is not self.target_func:
                return node
            new_body = []
            for stmt in node.body:
                new_stmt = self.__class__(self.target_func).visit(stmt)
                if isinstance(new_stmt, Sequence):
                    new_body.extend(new_stmt)
                else:
                    new_body.append(new_stmt)
            node.body = new_body
            ast.fix_missing_locations(node)
            return node

        def visit_Return(self, node):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                if node.value.func.id == self.target_func.name:
                    self.__class__.optimized = True
                    tmp_result = []
                    result = []

                    default_values = [] + self.target_func.args.defaults
                    for arg in self.target_func.args.args:
                        arg_name = ast.Name(id=arg.arg,
                                            ctx=ast.Store(),
                                            lineno=0,
                                            col_offset=0)
                        try:
                            arg_value = node.value.args.pop(0)
                        except IndexError:
                            arg_value = default_values.pop(0)
                        tmp_arg_sym = get_temp_name()
                        tmp_arg_name = ast.Name(id=tmp_arg_sym.name,
                                                ctx=ast.Store(),
                                                lineno=0,
                                                col_offset=0)
                        arg_value_assign = ast.Assign(targets=[tmp_arg_name],
                                                      value=arg_value,
                                                      lineno=0,
                                                      col_offset=0)
                        tmp_result.append(arg_value_assign)
                        arg_value_ref = translator.translate_ref(tmp_arg_sym)[1]
                        arg_re_assign = ast.Assign(targets=[arg_name],
                                                   value=arg_value_ref,
                                                   lineno=0,
                                                   col_offset=0)
                        arg_re_assign.re_assign = True
                        result.append(arg_re_assign)
                    if self.target_func.args.vararg is not None:
                        vararg = self.target_func.args.vararg
                        arg_name = ast.Name(id=vararg.arg if IS_PYTHON_34 else vararg,
                                            ctx=ast.Store(),
                                            lineno=0,
                                            col_offset=0)
                        arg_value = ast.Tuple(elts=node.value.args,
                                              lineno=0,
                                              col_offset=0,
                                              ctx=ast.Load())
                        arg_re_assign = ast.Assign(targets=[arg_name],
                                                   value=arg_value,
                                                   lineno=0,
                                                   col_offset=0)
                        arg_re_assign.re_assign = True
                        result.append(arg_re_assign)
                    result.append(ast.Continue(lineno=0, col_offset=0))
                    return tmp_result + result
            return node

    def _tail_recursion_optimize(self, func_ast):
        transformer = self.SelfTailRecursiveCallTransformer(func_ast)
        optimized_func_ast = transformer.visit(func_ast)

        if self.SelfTailRecursiveCallTransformer.optimized:
            if IS_PYPY:
                optimized_func_ast.body = [ast.While(test=ast.Name(id='True',
                                                                   ctx=ast.Load(),
                                                                   lineno=0,
                                                                   col_offset=0),
                                                     body=optimized_func_ast.body + [ast.Break(lineno=0,
                                                                                               col_offset=0)],
                                                     orelse=[],
                                                     lineno=0,
                                                     col_offset=0)]
            else:
                optimized_func_ast.body = [ast.While(test=ast.Num(n=1,
                                                                  lineno=0,
                                                                  col_offset=0),
                                                     body=optimized_func_ast.body + [ast.Break(lineno=0,
                                                                                               col_offset=0)],
                                                     orelse=[],
                                                     lineno=0,
                                                     col_offset=0)]
            self.SelfTailRecursiveCallTransformer.optimized = False
            return optimized_func_ast
        else:
            return func_ast

    def _tail_to_return_s(self, exps):

        def except_tail_to_return(except_exp):
            if issequence_except_str(except_exp) and isinstance(except_exp[0], Symbol)\
                    and except_exp[0].name == 'except':
                return (except_exp[0],) + self._tail_to_return_s(except_exp[1:])
            return except_exp

        # expは末尾の式。
        # expがシーケンスではないとき、returnに変換。
        # expがシーケンスで、先頭がreturn、yield、yield fromのとき、そのまま。
        # expがシーケンスで、先頭がif、do、try、matchのとき、中のシーケンスの末尾をreturnに変換。
        # expがシーケンスでそれ以外の場合、末尾の式をreturnに変換。
        exp = exps[-1]
        if not issequence_except_str(exp):
            exp = (Symbol('return'), exp)
        elif not isinstance(exp[0], Symbol):
            exp = (Symbol('return'), exp)
        elif exp[0].name in {'val', '_val', 'return', 'yield', 'yield-from', 'pass', 'raise'}:
            pass
        elif exp[0].name == 'if':
            if_exp = exp
            if len(if_exp) == 3:
                if_sym, test_cls, true_cls = if_exp
                exp = (if_sym, test_cls, self._tail_to_return_s([true_cls])[0])
            elif len(if_exp) == 4:
                if_sym, test_cls, true_cls, else_cls = if_exp
                exp = (if_sym, test_cls, self._tail_to_return_s([true_cls])[0], self._tail_to_return_s([else_cls])[0])
            else:
                new_if_exp = list(if_exp)
                i = 2
                exp_len = len(if_exp)
                while i < exp_len:
                    new_if_exp[i] = self._tail_to_return_s([if_exp[i]])[0]
                    i += 2
                if exp_len % 2 == 0:
                    new_if_exp[-1] = self._tail_to_return_s([if_exp[-1]])[0]
                exp = tuple(new_if_exp)
        elif exp[0].name in {'do', 'finally', 'except'}:
            exp = (exp[0], ) + self._tail_to_return_s(exp[1:])
        elif exp[0].name == 'try':
            exp = (exp[0], ) + self._tail_to_return_s(tuple(filter(lambda item: not (issequence_except_str(item)
                                                                                     and isinstance(item[0], Symbol)
                                                                                     and item[0].name in {'except',
                                                                                                          'finally'}),
                                                                   exp[1:]))) + \
                  tuple(map(except_tail_to_return, filter(lambda item: issequence_except_str(item) and
                                                                       isinstance(item[0], Symbol) and
                                                                       item[0].name in {'except', 'finally'}, exp[1:])))
        elif exp[0].name == 'match':
            new_exp = list(exp)
            i = 3
            exp_len = len(exp)
            while i < exp_len:
                new_exp[i] = self._tail_to_return_s([exp[i]])[0]
                i += 2
            exp = tuple(new_exp)
        else:
            exp = (Symbol('return'), exp)
        return tuple(exps[:-1]) + (exp,)

    @syntax('return')
    def translate_return(self, exp):
        if len(exp) != 2:
            raise MochiSyntaxError(exp, self.filename)

        pre, value = self.translate(exp[1], False)
        if type(value) is ast.Expr:
            value = value.value
        return pre + [ast.Return(value=value,
                                 lineno=exp[0].lineno,
                                 col_offset=0)], self.translate(NONE_SYM)[1]

    def _translate_handler(self, exp):
        type_expr, var, *body = exp
        return ast.ExceptHandler(type=self.translate(type_expr, False)[1],
                                 name=var.name,
                                 body=self._translate_sequence(body, True),
                                 lineno=var.lineno,
                                 col_offset=0)

    def _translate_handlers(self, exps):
        handlers = []
        for exp in exps:
            handlers.append(self._translate_handler(exp))
        return handlers

    @syntax('try')
    def translate_try(self, exp):
        if len(exp) < 2:
            raise MochiSyntaxError(exp, self.filename)
        body_exp = []
        handler_exps = []
        orelse_exp = []
        final_body_exp = []
        for expr in exp:
            if issequence_except_str(expr) and len(expr) > 1 and isinstance(expr[0], Symbol):
                expr_name = expr[0].name
                if expr_name == 'finally':
                    final_body_exp = expr[1:]
                    continue
                elif expr_name == 'orelse':
                    orelse_exp = expr[1:]
                    continue
                elif expr_name == 'except':
                    handler_exps.append(expr[1:])
                    continue
            body_exp.append(expr)

        body = self._translate_sequence(body_exp, True)
        handlers = self._translate_handlers(handler_exps)
        orelse = self._translate_sequence(orelse_exp, True)
        final_body = self._translate_sequence(final_body_exp, True)
        if IS_PYTHON_34:
            return (ast.Try(body=body,
                            handlers=handlers,
                            orelse=orelse,
                            finalbody=final_body,
                            lineno=exp[0].lineno,
                            col_offset=0),), self.translate(EMPTY_SYM, False)[1]
        else:
            if len(handlers) == 0:
                return (ast.TryFinally(body=body,
                                       finalbody=final_body,
                                       lineno=exp[0].lineno,
                                       col_offset=0),), self.translate(EMPTY_SYM, False)[1]
            else:
                return (ast.TryFinally(body=[ast.TryExcept(body=body,
                                                           handlers=handlers,
                                                           orelse=orelse,
                                                           lineno=exp[0].lineno,
                                                           col_offset=0)],
                                       finalbody=final_body,
                                       lineno=exp[0].lineno,
                                       col_offset=0),), self.translate(EMPTY_SYM, False)[1]

    @syntax('raise')
    def translate_raise(self, exp):
        if len(exp) != 2:
            raise MochiSyntaxError(exp, self.filename)
        return (ast.Raise(exc=self.translate(exp[1], False)[1],
                          cause=None,
                          lineno=exp[0].lineno,
                          col_offset=0),), self.translate(EMPTY_SYM, False)[1]

    @syntax('def')
    def translate_defun(self, exp, decorator_exps=None, visible=True):
        if not (len(exp) >= 4 and type(exp[1]) is Symbol):
            raise MochiSyntaxError(exp, self.filename)

        id = exp[1].name
        arg = exp[2]
        vararg_exp = None
        varkwarg_exp = None

        if not visible:
            self.hidden_vars.append(id)

        if type(arg) is Symbol:
            arg_exps = []
            vararg_exp = arg
        else:
            vararg_count = arg.count(VARG)
            varkwarg_count = arg.count(VKWARG)
            if vararg_count > 1 or varkwarg_count > 1:
                raise MochiSyntaxError(exp, self.filename)
            elif vararg_count and varkwarg_count:
                VARGIndex = arg.index(VARG)
                VKWARGIndex = arg.index(VKWARG)
                if (VARGIndex > VKWARGIndex):
                    raise MochiSyntaxError(exp, self.filename)
                arg_exps = arg[:VARGIndex]
                vararg_exp = arg[VARGIndex + 1]
                varkwarg_exp = arg[VKWARGIndex + 1:]
                if len(varkwarg_exp) == 1:
                    varkwarg_exp = varkwarg_exp[0]
                else:
                    raise MochiSyntaxError(exp, self.filename)
            elif vararg_count:
                VARGIndex = arg.index(VARG)
                arg_exps = arg[:VARGIndex]
                vararg_exp = arg[VARGIndex + 1:]
                if len(vararg_exp) == 1:
                    vararg_exp = vararg_exp[0]
                else:
                    raise MochiSyntaxError(exp, self.filename)
            elif varkwarg_count:
                VKWARGIndex = arg.index(VKWARG)
                arg_exps = arg[:VKWARGIndex]
                varkwarg_exp = arg[VKWARGIndex + 1:]
                if len(varkwarg_exp) == 1:
                    varkwarg_exp = varkwarg_exp[0]
                else:
                    raise MochiSyntaxError(exp, self.filename)
            else:
                arg_exps = arg

        args = self._translate_args(arg_exps)
        if vararg_exp is None:
            vararg = None
        else:
            vararg = ast.arg(arg=vararg_exp.name,
                             annotation=None) if IS_PYTHON_34 else vararg_exp.name

        if varkwarg_exp is None:
            varkwarg = None
        else:
            varkwarg = ast.arg(arg=varkwarg_exp.name,
                               annotation=None) if IS_PYTHON_34 else varkwarg_exp.name

        # 末尾のS式をreturnに変換する。
        # 末尾がif式やdo式の場合はその中まで変換する。
        exp = tuple(exp[:3]) + self._tail_to_return_s(exp[3:])  # TODO 最後が変数参照だと戻り値がNoneになる。
        body = self._translate_sequence(exp[3:])

        if varkwarg is not None:
            node = ast.parse('{0} = pmap({0})'.format(varkwarg.arg if IS_PYTHON_34 else varkwarg))
            body = node.body + body

        pre = []
        decorator_list = []
        if decorator_exps is not None:
            for decoratorExp in decorator_exps:
                pre_deco, value_deco = self.translate(decoratorExp, False)
                pre.extend(pre_deco)
                decorator_list.append(value_deco)

        defn = ast.FunctionDef(name=id,
                               args=ast.arguments(args=args,
                                                  vararg=vararg,
                                                  varargannotation=None,
                                                  kwonlyargs=[], kwarg=varkwarg,
                                                  kwargannotation=None,
                                                  defaults=[], kw_defaults=[],
                                                  lineno=exp[1].lineno,
                                                  col_offset=0),
                               body=body,
                               decorator_list=decorator_list,
                               returns=None,
                               lineno=exp[1].lineno,
                               col_offset=0)

        defn = self._tail_recursion_optimize(self._check_duplicated_binding(defn))
        pre.append(defn)
        _, ref = self.translate_ref(exp[1])
        return pre, ref

    @syntax('record')
    def translate_record(self, exp):
        exp_length = len(exp)
        if not (exp_length > 2 and isinstance(exp[1], Symbol)):
            raise MochiSyntaxError(exp, self.filename)

        if not ((isinstance(exp[2], Symbol) and issequence_except_str(exp[3])) or issequence_except_str(exp[2])):
            raise MochiSyntaxError(exp, self.filename)

        record_name = exp[1]  # Symbol
        parent_record = exp[2] if isinstance(exp[2], Symbol) else RECORD_SYM  # Symbol
        members_exp = exp[3] if isinstance(exp[2], Symbol) else exp[2]  # Sequence
        body_exps = exp[4:] if isinstance(exp[2], Symbol) else exp[3:]  # Sequence
        lineno = record_name.lineno
        col_offset = record_name.col_offset
        pre = []
        members = []
        body = [ast.Assign(targets=[ast.Name(id='__slots__',
                                             lineno=record_name.lineno,
                                             col_offset=record_name.col_offset,
                                             ctx=ast.Store())],
                           value=ast.Tuple(elts=[],
                                           lineno=record_name.lineno,
                                           col_offset=record_name.col_offset,
                                           ctx=ast.Load()),
                           lineno=record_name.lineno,
                           col_offset=record_name.col_offset)]

        # check_duplicated_binding_name_outer(record_name, self.filename)

        for member_exp in members_exp:
            member_pre, member = self.translate(member_exp.name, False)
            pre.extend(member_pre)
            members.append(member)

        for body_exp in body_exps:
            if isinstance(body_exp, (tuple, list)) and body_exp[0] == DEF:
                defun_pre, _ = self.translate_defun(body_exp)
                pre.extend(defun_pre[:-1])
                body.append(defun_pre[-1])
            else:
                body_pre, body_py_exp = self.translate(body_exp)
                pre.extend(body_pre)
                body.append(body_py_exp)

        record_def = ast.ClassDef(name=record_name.name,
                                  bases=[ast.Call(func=ast.Name(id='pclass',
                                                                lineno=lineno,
                                                                col_offset=col_offset,
                                                                ctx=ast.Load()),
                                                  args=[
                                                      ast.BinOp(left=ast.Attribute(value=ast.Name(id=parent_record.name,
                                                                                                  lineno=parent_record.lineno,
                                                                                                  col_offset=parent_record.col_offset,
                                                                                                  ctx=ast.Load()),
                                                                                   attr='_fields',
                                                                                   lineno=parent_record.lineno,
                                                                                   col_offset=parent_record.col_offset,
                                                                                   ctx=ast.Load()),
                                                                op=ast.Add(),
                                                                right=ast.Tuple(elts=members,
                                                                                lineno=parent_record.lineno,
                                                                                col_offset=parent_record.col_offset,
                                                                                ctx=ast.Load()),
                                                                lineno=parent_record.lineno,
                                                                col_offset=parent_record.col_offset),
                                                      ast.Str(s=record_name.name,
                                                              lineno=lineno,
                                                              col_offset=col_offset)],
                                                  keywords=[],
                                                  starargs=None,
                                                  kwargs=None,
                                                  lineno=parent_record.lineno,
                                                  col_offset=parent_record.col_offset),
                                         ast.Name(id=parent_record.name,
                                                  lineno=parent_record.lineno,
                                                  col_offset=parent_record.col_offset,
                                                  ctx=ast.Load())],
                                  keywords=[],
                                  starargs=None,
                                  kwargs=None,
                                  lineno=lineno,
                                  col_offset=col_offset,
                                  body=body,
                                  decorator_list=[])

        pre.append(record_def)
        _, def_ref = self.translate_ref(record_name)
        record_ids.append(record_name.name)
        return pre, def_ref

    @syntax('class')
    def translate_class(self, exp):
        exp_length = len(exp)
        if not (exp_length > 2 and isinstance(exp[1], Symbol)):
            raise MochiSyntaxError(exp, self.filename)

        if not ((isinstance(exp[2], Symbol) and issequence_except_str(exp[3])) or issequence_except_str(exp[2])):
            raise MochiSyntaxError(exp, self.filename)

        class_name = exp[1]  # Symbol
        parent_class = exp[2] if isinstance(exp[2], Symbol) else OBJECT_SYM  # Symbol
        members_exp = exp[3] if isinstance(exp[2], Symbol) else exp[2]  # Sequence
        body_exps = exp[4:] if isinstance(exp[2], Symbol) else exp[3:]  # Sequence
        lineno = class_name.lineno
        col_offset = class_name.col_offset
        pre = []
        members = []

        for member_exp in members_exp:
            member_pre, member = self.translate(member_exp.name, False)
            pre.extend(member_pre)
            members.append(member)

        body = [ast.Assign(targets=[ast.Name(id='__slots__',
                                             lineno=lineno,
                                             col_offset=col_offset,
                                             ctx=ast.Store())],
                           value=ast.Tuple(elts=members,
                                           lineno=lineno,
                                           col_offset=col_offset,
                                           ctx=ast.Load()),
                           lineno=class_name.lineno,
                           col_offset=class_name.col_offset)]

        # check_duplicated_binding_name_outer(class_name, self.filename) # TODO

        for body_exp in body_exps:
            if isinstance(body_exp, (tuple, list)) and body_exp[0] == DEF:
                defun_pre, _ = self.translate_defun(body_exp)
                pre.extend(defun_pre[:-1])
                body.append(defun_pre[-1])
            else:
                body_pre, body_py_exp = self.translate(body_exp)
                pre.extend(body_pre)
                body.append(body_py_exp)

        class_def = ast.ClassDef(name=class_name.name,
                                 bases=[ast.Name(id=parent_class.name,
                                                 lineno=parent_class.lineno,
                                                 col_offset=parent_class.col_offset,
                                                 ctx=ast.Load())],
                                 keywords=[],
                                 starargs=None,
                                 kwargs=None,
                                 lineno=lineno,
                                 col_offset=col_offset,
                                 body=body,
                                 decorator_list=[])
        pre.append(class_def)
        _, def_ref = self.translate_ref(class_name)
        return pre, def_ref

    @syntax('import')
    def translate_import(self, exp):
        if len(exp) < 2:
            raise MochiSyntaxError(exp, self.filename)
        names = [ast.alias(name=import_sym.name,  # check_duplicated_binding_name(import_sym, self.filename).name,
                           asname=None,
                           lineno=import_sym.lineno,
                           col_offset=import_sym.col_offset) for import_sym in exp[1:]]
        return (ast.Import(names=names,
                           lineno=exp[0].lineno,
                           col_offset=exp[0].col_offset),), self.translate_ref(NONE_SYM)[1]

    @syntax('from-import')
    def translate_from(self, exp):
        if len(exp) < 3:
            raise MochiSyntaxError(exp, self.filename)
        names = [ast.alias(name=import_sym.name,
                           asname=None,
                           lineno=import_sym.lineno,
                           col_offset=import_sym.col_offset) for import_sym in exp[2:]]
        return (ast.ImportFrom(module=exp[1].name,
                               names=names,
                               lineno=exp[0].lineno,
                               col_offset=exp[0].col_offset),), self.translate_ref(NONE_SYM)[1]

    def is_macro(self, exp):
        if not issequence_except_str(exp):
            return False
        if not isinstance(exp[0], Symbol):
            return False
        return translator.has_macro(exp[0].name)

    def has_macro(self, name):
        return name in self.macro_table

    def add_macro(self, name, macro_func):
        self.macro_table[name] = macro_func

    def expand_macro(self, name, args):
        if name not in global_env:
            code = compile(
                ast.Module(body=[self.macro_table[name]]), self.filename, 'exec', optimize=2)
            exec(code, global_env)
        macro_func = global_env[name]
        keyword_params = {}
        newargs = []
        for arg in args:
            # if type(arg) is Keyword:
            #    keyword_params = arg
            #    continue
            newargs.append(arg)
            args = newargs
        return tuple_it(macro_func(*args, **keyword_params))

    @syntax('mac')
    def translate_def_macro(self, exp):
        macro_func_ast_seq, ref = self.translate_defun(exp)
        self.add_macro(exp[1].name, macro_func_ast_seq[0])
        return macro_func_ast_seq, ref

    def translate_apply(self, exp):
        pre = []
        callable_pre, callable_value = self.translate(exp[0], False)
        pre.extend(callable_pre)

        args = []
        keyword_args = []
        keyword_arg_exps = []
        arg_exps = exp[1:]
        for i, argexp in enumerate(arg_exps):
            if type(argexp) is Keyword:
                keyword_arg_exps = arg_exps[i:]
                arg_exps = arg_exps[:i]
                break

        for argexp in arg_exps:
            arg_pre, arg_value = self.translate(argexp, False)
            pre.extend(arg_pre)
            args.append(arg_value)

        for argKey, argExp in chunks(keyword_arg_exps, 2):
            if type(argKey) is not Keyword:
                raise MochiSyntaxError(argKey, self.filename)
            arg_pre, arg_value = self.translate(argExp, False)
            pre.extend(arg_pre)
            keyword_args.append(ast.keyword(arg=argKey.name,
                                            value=arg_value))

        value = ast.Call(func=callable_value,
                         args=args,
                         keywords=keyword_args,
                         starargs=None,
                         kwargs=None,
                         lineno=callable_value.lineno,
                         col_offset=0)
        return pre, value

    @syntax('fn')
    def translate_fn(self, exp):
        if not (len(exp) >= 3):
            raise MochiSyntaxError(exp, self.filename)
        name_symbol = get_temp_name()
        defexp = []
        defexp.append('dummy')
        defexp.append(name_symbol)
        defexp.extend(exp[1:])
        return self.translate_defun(defexp, visible=False)

    @syntax('make-tuple')
    @syntax('make-pvector')
    def translate_make_pvector(self, exp):
        make_tuple_symbol, *items = exp
        pre = []
        elts = []
        lineno = make_tuple_symbol.lineno
        col_offset = make_tuple_symbol.col_offset
        for item in items:
            item_pre, item_value = self.translate(item, False)
            pre.extend(item_pre)
            elts.append(item_value)
        tuple_ast = ast.Tuple(elts=elts,
                              ctx=ast.Load(),
                              lineno=lineno,
                              col_offset=col_offset)
        return pre, ast.Call(func=ast.Name(id='pvector',
                                           ctx=ast.Load(),
                                           lineno=lineno,
                                           col_offset=col_offset),
                             args=[tuple_ast],
                             keywords=[],
                             starargs=None,
                             kwargs=None,
                             lineno=lineno,
                             col_offset=col_offset)

    @syntax('make-list')
    def translate_make_list(self, exp):
        make_list_symbol, *items = exp
        pre = []
        elts = []
        lineno = make_list_symbol.lineno
        col_offset = make_list_symbol.col_offset
        for item in items:
            item_pre, item_value = self.translate(item, False)
            pre.extend(item_pre)
            elts.append(item_value)
        return pre, ast.List(elts=elts,
                             ctx=ast.Load(),
                             lineno=lineno,
                             col_offset=col_offset)

    @syntax('quote')
    def translate_quote(self, exp):
        if len(exp) != 2:
            raise MochiSyntaxError(exp, self.filename)
        quote_symbol, value = exp
        if issequence_except_str(value):
            pre = []
            elts = []
            lineno = quote_symbol.lineno
            col_offset = quote_symbol.col_offset
            for item in value:
                item_pre, item_value = self.translate(item, False, True)
                pre.extend(item_pre)
                elts.append(item_value)
            return pre, ast.Tuple(elts=elts,
                                  ctx=ast.Load(),
                                  lineno=lineno,
                                  col_offset=col_offset)
        elif isinstance(value, Symbol):
            lineno = value.lineno
            col_offset = value.col_offset
            return (EMPTY, ast.Call(func=ast.Name(id='Symbol',
                                                  ctx=ast.Load(),
                                                  lineno=lineno,
                                                  col_offset=col_offset),
                                    args=[ast.Str(s=value.name,
                                                  lineno=lineno,
                                                  col_offset=col_offset)],
                                    keywords=[],
                                    starargs=None,
                                    kwargs=None,
                                    lineno=lineno,
                                    col_offset=col_offset))
        else:
            return self.translate_atom(value)

    def call_unquote_splicing(self, list_ast):
        lineno = list_ast.lineno
        col_offset = list_ast.col_offset
        return ast.Call(func=ast.Name(id='unquote_splice',
                                      ctx=ast.Load(),
                                      lineno=lineno,
                                      col_offset=col_offset),
                        args=[list_ast],
                        keywords=[],
                        starargs=None,
                        kwargs=None,
                        lineno=lineno,
                        col_offset=col_offset)

    @syntax('quasiquote')
    def translate_quasi_quote(self, exp):
        if len(exp) != 2:
            raise MochiSyntaxError(exp, self.filename)
        quote_symbol, value = exp
        if issequence_except_str(value):
            if len(value) >= 1 and type(value[0]) is Symbol:
                if value[0].name == 'unquote':
                    return self.translate(value[1], False)
                elif value[0].name == 'unquote-splicing':
                    value1_pre, value1_body = self.translate(value[1], False)
                    return (tuple(value1_pre),
                            [self.translate((Symbol('quote'),
                                             Symbol('_mochi_unquote-splicing')), False), value1_body])
            lineno = quote_symbol.lineno
            col_offset = quote_symbol.col_offset
            pre = []
            elts = []
            for item in value:
                item_pre, item_value = self.translate(item, True, False, True)
                pre.extend(item_pre)
                elts.append(item_value)
            return (pre, self.call_unquote_splicing(ast.Tuple(elts=flatten_list(elts),
                                                              ctx=ast.Load(),
                                                              lineno=lineno,
                                                              col_offset=col_offset)))
        elif isinstance(value, Symbol):
            lineno = value.lineno
            col_offset = value.col_offset
            return (EMPTY, ast.Call(func=ast.Name(id='Symbol',
                                                  ctx=ast.Load(),
                                                  lineno=lineno,
                                                  col_offset=col_offset),
                                    args=[ast.Str(s=value.name,
                                                  lineno=lineno,
                                                  col_offset=col_offset)],
                                    keywords=[],
                                    starargs=None,
                                    kwargs=None,
                                    lineno=lineno,
                                    col_offset=col_offset))
        else:
            return self.translate_atom(value)

    def enclose(self, py_ast, flag):

        if isinstance(py_ast, expr_and_stmt_ast):
            return py_ast

        if issequence_except_str(py_ast):
            ast_list = []
            for item in py_ast:
                if isinstance(item, expr_and_stmt_ast):
                    ast_list.append(item)
                elif issequence_except_str(item):
                    # ((pre_foo ...), value_bar) => ((enclose(pre_foo) ...), value_bar)
                    newitem = []
                    for itemitem in item:
                        if isinstance(itemitem, expr_and_stmt_ast):
                            newitem.append(itemitem)
                        else:
                            newitem.append(
                                ast.Expr(value=itemitem, lineno=itemitem.lineno, col_offset=0))
                    ast_list.append(newitem)
                else:
                    if flag:
                        ast_list.append(ast.Expr(value=item, lineno=item.lineno, col_offset=0))
                    else:
                        ast_list.append(item)
            return ast_list
        else:
            if flag and (not isinstance(py_ast, expr_and_stmt_ast)):
                return ast.Expr(value=py_ast, lineno=py_ast.lineno, col_offset=py_ast.col_offset)
            else:
                return py_ast

    def translate(self, exp, enclose=True, quoted=False, quasi_quoted=False):
        if quoted:
            quoted_exp = [Symbol('quote'), exp]
            return self.translate_quote(quoted_exp)

        if quasi_quoted:
            quoted_exp = [Symbol('quote'), exp]
            return self.translate_quasi_quote(quoted_exp)

        if not issequence_except_str(exp):
            return self.enclose(self.translate_atom(exp), enclose)

        if type(exp[0]) is Symbol:
            if exp[0].name == 'load':
                tl = Translator(self.macro_table)
                return tl.translate_loaded_files(exp[1:])
            elif exp[0].name == 'require':
                tl = Translator(self.macro_table)
                return tl.translate_required_files(exp[1:])
            elif exp[0].name in self.macro_table:
                return self.translate(self.expand_macro(exp[0].name, exp[1:]), enclose)
            elif exp[0].name in syntax_table:
                return self.enclose(syntax_table[exp[0].name](self, exp), enclose)
        return self.enclose(self.translate_apply(exp), enclose)


def interact():
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
                break

            try:
                lexer = lex(buffer, repl_mode=True)

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
            except Exception as e:
                print(e)
                continuation_flag = False
                buffer = ''
                continue

        try:
            sexps = parse(tokens.__iter__())
            for sexp in sexps:
                if isinstance(sexp, MutableSequence):
                    sexp = tuple_it(sexp)
                if sexp is COMMENT:
                    continue
                py_ast = translator.translate_sexp_to_interact(sexp)
                if py_ast is not None:
                    code = compile(py_ast, '<string>', 'exec')
                    if code is not None:
                        exec(code, global_env)
                    py_print()
        except Exception as e:
            # traceback.print_tb(sys.exc_info()[2])
            print('*** ERROR: ' + str(e))


py_eval = eval


@builtin
def eval(str):
    sr = SexpReader(InputPort(StringIO(str)))
    while True:
        sexp = sr.get_sexp()

        if sexp is None:
            break
        if sexp is EOF:
            break
        if sexp is COMMENT:
            continue

        py_ast = translator.translate_sexp(tuple_it(sexp))

        if py_ast is not None:
            code = compile(py_ast, '<string>', 'exec')
            if code is not None:
                exec(code, global_env)


@builtin_rename('_del-hidden-var')
def del_hidden_var(name):
    translator.hidden_vars.remove(name)


@builtin_rename('_get-hidden-vars')
def get_hidden_vars():
    return translator.hidden_vars


@builtin_rename('delitem')
def _del(name, env):
    del env[name]


def output_pyc(code):
    import marshal
    import time
    import io

    pyc_bytes = io.BytesIO()
    pyc_bytes.write(b'\0\0\0\0')
    wr_long(pyc_bytes, int(time.time()))
    size = 0  # TODO
    wr_long(pyc_bytes, size)
    marshal.dump(code, pyc_bytes)
    pyc_bytes.flush()
    pyc_bytes.seek(0, 0)
    pyc_bytes.write(MAGIC)
    pyc_bytes.seek(0, 0)
    sys.stdout.buffer.write(pyc_bytes.read())


def output_code(code):
    import marshal

    marshal.dump(code, sys.stdout.buffer)


def compile_file(src_path, optimize=-1):
    # binding_name_set_stack[0].update(global_env.keys())
    py_ast = translator.translate_file(src_path)
    return compile(py_ast, src_path, 'exec', optimize=optimize)


def load_file(path, env):
    return exec(compile_file(path), env)


def execute_compiled_file(path):
    import marshal

    with open(path, 'rb') as compiled_file:
        return exec(marshal.load(compiled_file), global_env)


#@builtin
#def load(path):
#    load_file(path, globalEnv)


@builtin_rename('macex1')
@builtin_rename('macroexpand-1')
def macroexpand1(exp):
    if translator.is_macro(exp):
        return translator.expand_macro(exp[0].name, exp[1:])
    return exp


@builtin
@builtin_rename('macex')
def macroexpand(exp):
    if translator.is_macro(exp):
        return macroexpand(macroexpand1(exp))
    return exp


@builtin_rename('is-macro')
def is_macro(func):
    return isinstance(func, FunctionType) and translator.has_macro(func.__name__)


@builtin_rename('is-function')
def is_function(func):
    return isinstance(func, FunctionType) and (not translator.has_macro(func.__name__))


@builtin_rename('iter-except')
def iter_except(func, exc):
    try:
        while 1:
            yield func()
    except exc:
        pass


@builtin_rename('assert')
def _assert(value):
    assert value


@builtin_rename('assert-equal')
def assert_equal(value1, value2):
    assert value1 == value2


@builtin_rename('assert-not-equal')
def assert_not_equal(value1, value2):
    assert value1 != value2


@builtin_rename('assert-none')
def assert_none(value):
    assert value is None


@builtin_rename('assert-not-none')
def assert_not_none(value):
    assert value is not None


@builtin_rename('assert-same')
def assert_same(value1, value2):
    assert value1 is value2


@builtin_rename('assert-not-same')
def assert_not_same(value1, value2):
    assert value1 is not value2


translator = Translator()
global_scope = Scope()
for name in global_env.keys():
    global_scope.add_binding_name(name, "<builtin>")
binding_name_set_stack = [global_scope]


#if __name__ == '__main__':
def main():
    # 'translator' and 'bindingNameSetStack' is a global variable.

    eval("""
(mac get! (ref)
  `((get ,ref 0)))

(def flatten (list-of-lists)
  (chain.from_iterable list-of-lists))

(mac set! (ref update-func)
 `((get ,ref 1) ,update-func))

(mac val (pattern value)
  (def keyword->str (keyword)
    (if (isinstance keyword Keyword)
      (str keyword)
      keyword))

  (def mapping-match? (target pattern)
    (_val len-pattern (len pattern))
    (if
      (== len-pattern 0) 'True
      (== len-pattern 1) `(in ,(keyword->str (car pattern)) ,target)
      (== len-pattern 2) `(and (in ,(keyword->str (car pattern)) ,target)
       ,(match? (v 'get target (keyword->str (car pattern))) (cadr pattern)))
      `(and ,(mapping-match? target (get pattern 0 2)) ,(mapping-match? target (get pattern 2 None)))))

  (def match? (target pattern)
      (if (== pattern 'True) `(is ,target True)
          (== pattern 'False) `(is ,target False)
          (== pattern 'None) `(is ,target None)
          (== pattern '_) 'True
          (isinstance pattern Symbol) 'True
          (isinstance pattern tuple)
            (do
              (_val len-pattern (len pattern))
                (if
                  (== len-pattern 0) `(and (isinstance ,target Sequence) (== (len ,target) 0))
                  (record-id? (str (car pattern))) `(and (isinstance ,target ,(car pattern))
                                                   ,(match? target (cdr pattern)))
                  (== (car pattern) 'make-list) (match? target (cdr pattern))
                  (== (car pattern) 'make-tuple) (match? target (cdr pattern))
                  (== (car pattern) 'table)
                    (if
                      (== len-pattern 1) `(isinstance ,target Mapping)
                      (> len-pattern 1) `(and (isinstance ,target Mapping) ,(mapping-match? target (cdr pattern)))
                      'False)
                  (== (car pattern) 'dict*)
                    (if
                      (== len-pattern 1) `(isinstance ,target Mapping)
                      (> len-pattern 1) `(and (isinstance ,target Mapping) ,(mapping-match? target (cdr pattern)))
                      'False)
                  (== (car pattern) 'bool)
                    (if
                      (== len-pattern 1) `(isinstance ,target bool)
                      (== len-pattern 2) `(and (isinstance ,target bool) ,(match? target (cadr pattern)))
                      'False)
                  (== (car pattern) 'int)
                    (if
                      (== len-pattern 1) `(isinstance ,target int)
                      (== len-pattern 2) `(and (isinstance ,target int) ,(match? target (cadr pattern)))
                      'False)
                  (== (car pattern) 'float)
                    (if
                      (== len-pattern 1) `(isinstance ,target float)
                      (== len-pattern 2) `(and (isinstance ,target float) ,(match? target (cadr pattern)))
                      'False)
                  (== (car pattern) 'Number)
                    (if
                      (== len-pattern 1) `(isinstance ,target Number)
                      (== len-pattern 2) `(and (isinstance ,target Number) ,(match? target (cadr pattern)))
                      'False)
                  (== (car pattern) 'str)
                    (if
                      (== len-pattern 1) `(isinstance ,target str)
                      (== len-pattern 2) `(and (isinstance ,target str) ,(match? target (cadr pattern)))
                      'False)
                  (and (== (car pattern) 'not) (== len-pattern 2))
                    `(not ,(match? target (cadr pattern)))
                  (== (car pattern) 'or)
                    (if
                      (== len-pattern 1) 'False
                      (== len-pattern 2) (match? target (cadr pattern))
                      `(or ,(match? target (cadr pattern)) ,(match? target (cons 'or (cddr pattern)))))
                  (== (car pattern) 'and)
                    (if
                      (== len-pattern 1) 'False
                      (== len-pattern 2) (match? target (cadr pattern))
                      `(and ,(match? target (cadr pattern)) ,(match? target (cons 'and (cddr pattern)))))
                  (== (car pattern) 'fn) `(,pattern ,target)
                  (== (car pattern) 'quote) `(== ,pattern ,target)
                  (== (car pattern) 'type) `(isinstance ,target ,(get pattern 1))
                  (== len-pattern 1) `(and (isinstance ,target Sequence) (== (len ,target) 1) ,(match? (v 'car target)
                                                                                                     (car pattern)))
                  (in (Symbol "&") pattern) (do
                                              (_val len-pattern-fixed (pattern.index (Symbol "&")))
                                              `(and (isinstance ,target Sequence) (>= (len ,target) ,len-pattern-fixed)
                                                    ,(match? (v 'get target 0 len-pattern-fixed)
                                                             (get pattern 0 len-pattern-fixed))))
                  `(and (isinstance ,target Sequence) (== (len ,target) ,len-pattern)
                                                   ,(match? (v 'car target) (car pattern))
                                                   ,(match? (v 'cdr target) (cdr pattern)))))
          `(== ,target ,pattern)))

    (_val sym-num-seq (ref 0))
    (def gensym-match ()
      (set! sym-num-seq |+ _ 1|)
      (Symbol (+ "m" (str #!sym-num-seq))))

  (def table-pattern-bind (pattern target)
      (_val len-pattern (len pattern))
      (if
        (== len-pattern 0) '()
        (== len-pattern 1) '()
        (+ (pattern-bind (get pattern 1) `(get ,target ,(keyword->str (car pattern))))
           (table-pattern-bind (get pattern 2 None) target))))

  (def pattern-bind (pattern target)
      (if (isinstance pattern Symbol) `((_val ,pattern ,target))
          (isinstance pattern tuple)
            (do
              (_val len-pattern (len pattern))
              (if
                (== len-pattern 0) '()
                (== (car pattern) 'make-list) (pattern-bind (cdr pattern) target)
                (== (car pattern) 'make-tuple) (pattern-bind (cdr pattern) target)
                (== (car pattern) 'table) (table-pattern-bind (cdr pattern) target)
                (== (car pattern) 'dict*) (table-pattern-bind (cdr pattern) target)
                (== len-pattern 1) (pattern-bind (car pattern) `(car ,target))
                (in (Symbol "&") pattern) (do
                                            (_val len-pattern-fixed (pattern.index (Symbol "&")))
                                            (+ (pattern-bind (get pattern 0 len-pattern-fixed)
                                                             `(get ,target 0 ,len-pattern-fixed))
                                               (pattern-bind (get pattern (+ len-pattern-fixed 1))
                                                             `(get ,target ,len-pattern-fixed None))))
                (+ (pattern-bind (get pattern 0) `(car ,target))
                   (pattern-bind (get pattern 1 None) `(cdr ,target)))))
          '()))
  (if (match? value pattern)
    `(do ,@(pattern-bind pattern value))
    None))

(mac tuple-of (& form)
  (val (bodyexpr bindingform) form)
  (if (== (len bindingform) 0)
      `(v ,bodyexpr)
    (do
	  (val (binding seqexpr & bindings) bindingform)
	  (if (== binding ':when)
	    `(if ,seqexpr (tuple-of ,bodyexpr ,bindings))
	    `(mapcat (fn (,binding) (tuple-of ,bodyexpr ,bindings))
		         ,seqexpr)))))


(mac list-of (& form)
  (val (bodyexpr bindingform) form)
  (if (== (len bindingform) 0)
      `(list* ,bodyexpr)
    (do
	  (val (binding seqexpr & bindings) bindingform)
	  (if (== binding ':when)
	    `(if ,seqexpr (list-of ,bodyexpr ,bindings))
	    `(mapcat (fn (,binding) (list-of ,bodyexpr ,bindings))
		         ,seqexpr)))))


(mac let (args & body)
 `((fn ()
	,@(doall (map |cons 'val _| args))
	,@body)))

;(mac and args
;  (if (null args) True
;    (null (rest args)) (first args)
;    `(if ,(first args) (and ,@(rest args)) False)))

;(mac or args
;  (if (null args)
;    False
;    (if (null (rest args))
;      (first args)
;	  (let ((value (gensym)))
;	  `(let ((,value ,(first args)))
;	     (if ,value ,value (or ,@(rest args))))))))

(mac w/uniq (name & body)
  `(let ((,name (uniq)))
    ,@body))

(mac accum (accfn & body)
  (w/uniq gacc
    `(let ((,gacc #())
           (,accfn |append ,gacc _|))
      ,@body
      (tuple ,gacc))))

(def readlines (path)
  (gen-with (open path "r")
     (fn (lines) lines)))

(def writelines (path lines)
  (with (open path "w")
    (fn (f) (f.writelines lines))))

(mac defseq (name iterable)
  `(_val ,name (lazyseq ,iterable)))

(mac -> (operand & operators)
  (if (== (len operators) 0)
      operand
      (let ((operator (first operators))
	    (rest-operators (rest operators)))
	(if (isinstance operator tuple)
	    `(-> (,(first operator) ,operand ,@(rest operator)) ,@rest-operators)
	  `(-> (,operator ,operand) ,@rest-operators)))))

(mac ->> (operand & operators)
  (if (== (len operators) 0)
      operand
      (let ((operator (first operators))
	    (rest-operators (rest operators)))
	(if (isinstance operator tuple)
	    `(->> (,(first operator) ,@(rest operator) ,operand) ,@rest-operators)
	  `(->> (,operator ,operand) ,@rest-operators)))))

;(mac import (& targets)
;  `(_require_py_module (quote ,targets)))


(mac _match (target & pattern-procs)
    (def keyword->str (keyword)
      (if (isinstance keyword Keyword)
        (str keyword)
        keyword))

    (def mapping-match? (target pattern)
      (_val len-pattern (len pattern))
      (if
        (== len-pattern 0) 'True
        (== len-pattern 1) `(in ,(keyword->str (car pattern)) ,target)
        (== len-pattern 2) `(and (in ,(keyword->str (car pattern)) ,target)
         ,(match? (v 'get target (keyword->str (car pattern))) (cadr pattern)))
        `(and ,(mapping-match? target (get pattern 0 2)) ,(mapping-match? target (get pattern 2 None)))))

    (def match? (target pattern)
      (if (== pattern 'True) `(is ,target True)
          (== pattern 'False) `(is ,target False)
          (== pattern 'None) `(is ,target None)
          (== pattern '_) 'True
          (isinstance pattern Symbol) 'True
          (isinstance pattern tuple)
            (do
              (_val len-pattern (len pattern))
                (if
                  (== len-pattern 0) `(and (isinstance ,target Sequence) (== (len ,target) 0))
                  (record-id? (str (car pattern))) `(and (isinstance ,target ,(car pattern))
                                                   ,(match? target (cdr pattern)))
                  (== (car pattern) 'make-list) (match? target (cdr pattern))
                  (== (car pattern) 'make-tuple) (match? target (cdr pattern))
                  (== (car pattern) 'table)
                    (if
                      (== len-pattern 1) `(isinstance ,target Mapping)
                      (> len-pattern 1) `(and (isinstance ,target Mapping) ,(mapping-match? target (cdr pattern)))
                      'False)
                  (== (car pattern) 'dict*)
                    (if
                      (== len-pattern 1) `(isinstance ,target Mapping)
                      (> len-pattern 1) `(and (isinstance ,target Mapping) ,(mapping-match? target (cdr pattern)))
                      'False)
                  (== (car pattern) 'bool)
                    (if
                      (== len-pattern 1) `(isinstance ,target bool)
                      (== len-pattern 2) `(and (isinstance ,target bool) ,(match? target (cadr pattern)))
                      'False)
                  (== (car pattern) 'int)
                    (if
                      (== len-pattern 1) `(isinstance ,target int)
                      (== len-pattern 2) `(and (isinstance ,target int) ,(match? target (cadr pattern)))
                      'False)
                  (== (car pattern) 'float)
                    (if
                      (== len-pattern 1) `(isinstance ,target float)
                      (== len-pattern 2) `(and (isinstance ,target float) ,(match? target (cadr pattern)))
                      'False)
                  (== (car pattern) 'Number)
                    (if
                      (== len-pattern 1) `(isinstance ,target Number)
                      (== len-pattern 2) `(and (isinstance ,target Number) ,(match? target (cadr pattern)))
                      'False)
                  (== (car pattern) 'str)
                    (if
                      (== len-pattern 1) `(isinstance ,target str)
                      (== len-pattern 2) `(and (isinstance ,target str) ,(match? target (cadr pattern)))
                      'False)
                  (and (== (car pattern) 'not) (== len-pattern 2))
                    `(not ,(match? target (cadr pattern)))
                  (== (car pattern) 'or)
                    (if
                      (== len-pattern 1) 'False
                      (== len-pattern 2) (match? target (cadr pattern))
                      `(or ,(match? target (cadr pattern)) ,(match? target (cons 'or (cddr pattern)))))
                  (== (car pattern) 'and)
                    (if
                      (== len-pattern 1) 'False
                      (== len-pattern 2) (match? target (cadr pattern))
                      `(and ,(match? target (cadr pattern)) ,(match? target (cons 'and (cddr pattern)))))
                  (== (car pattern) 'fn) `(,pattern ,target)
                  (== (car pattern) 'quote) `(== ,pattern ,target)
                  (== (car pattern) 'type) `(isinstance ,target ,(get pattern 1))
                  (== len-pattern 1) `(and (isinstance ,target Sequence) (== (len ,target) 1) ,(match? (v 'car target)
                                                                                                     (car pattern)))
                  (in (Symbol "&") pattern) (do
                                              (_val len-pattern-fixed (pattern.index (Symbol "&")))
                                              `(and (isinstance ,target Sequence) (>= (len ,target) ,len-pattern-fixed)
                                                    ,(match? (v 'get target 0 len-pattern-fixed)
                                                             (get pattern 0 len-pattern-fixed))))
                  `(and (isinstance ,target Sequence) (== (len ,target) ,len-pattern)
                                                   ,(match? (v 'car target) (car pattern))
                                                   ,(match? (v 'cdr target) (cdr pattern)))))
          `(== ,target ,pattern)))

    (_val sym-num-seq (ref 0))
    (def gensym-match ()
      (set! sym-num-seq |+ _ 1|)
      (Symbol (+ "m" (str #!sym-num-seq))))

    (def table-pattern-bind (pattern target)
      (_val len-pattern (len pattern))
      (if
        (== len-pattern 0) '()
        (== len-pattern 1) '()
        (+ (pattern-bind (get pattern 1) `(get ,target ,(keyword->str (car pattern))))
           (table-pattern-bind (get pattern 2 None) target))))

    (def pattern-bind (pattern target)
      (if (== pattern 'True) '()
          (== pattern 'False) '()
          (== pattern 'None) '()
          (== pattern '_) '()
          (isinstance pattern Symbol) `((_val ,pattern ,target))
          (isinstance pattern tuple)
            (do
              (_val len-pattern (len pattern))
              (if
                (== len-pattern 0) '()
                (== (car pattern) 'not) '()
                (== (car pattern) 'or) '()
                (== (car pattern) 'and)
                  (if (== len-pattern 2)
                    (pattern-bind (cadr pattern) target)
                    (+ (pattern-bind (cadr pattern) target)
                       (pattern-bind (cons 'and (cddr pattern)) target)))
                (record-id? (str (car pattern))) (pattern-bind (cdr pattern) target)
                (== (car pattern) 'make-list) (pattern-bind (get pattern 1 None) target)
                (== (car pattern) 'make-tuple) (pattern-bind (get pattern 1 None) target)
                (== (car pattern) 'table) (if (> len-pattern 2) (table-pattern-bind (cdr pattern) target))
                (== (car pattern) 'dict*) (if (> len-pattern 2) (table-pattern-bind (cdr pattern) target))
                (== (car pattern) 'bool) (if (> len-pattern 1) (pattern-bind (cadr pattern) target))
                (== (car pattern) 'int) (if (> len-pattern 1) (pattern-bind (cadr pattern) target))
                (== (car pattern) 'float) (if (> len-pattern 1) (pattern-bind (cadr pattern) target))
                (== (car pattern) 'Number) (if (> len-pattern 1) (pattern-bind (cadr pattern) target))
                (== (car pattern) 'str) (if (> len-pattern 1) (pattern-bind (cadr pattern) target))
                (== (car pattern) 'fn) `((_val ,(gensym-match) ,target))
                (== (car pattern) 'quote) (pattern-bind (get pattern 1) target)
                (== (car pattern) 'type) `((_val ,(gensym-match) ,target))
                (== len-pattern 1) (pattern-bind (car pattern) `(car ,target))
                (in (Symbol "&") pattern) (do
                                            (_val len-pattern-fixed (pattern.index (Symbol "&")))
                                            (+ (pattern-bind (get pattern 0 len-pattern-fixed)
                                                             `(get ,target 0 ,len-pattern-fixed))
                                               (pattern-bind (get pattern (+ len-pattern-fixed 1))
                                                             `(get ,target ,len-pattern-fixed None))))
                (+ (pattern-bind (get pattern 0) `(car ,target))
                   (pattern-bind (get pattern 1 None) `(cdr ,target)))))
          '()))

     (_val targetval (gensym))
     (_val result (gensym))
     (_val len-pattern-procs (len pattern-procs))
     (if (== len-pattern-procs 0) 'False
       (>= len-pattern-procs 1)
       (let (((pattern & procs) (first pattern-procs)))
     `(do
	    (_val ,targetval ,target)
	    (_val ,result ,(match? targetval pattern))
	    (if ,result
		  (do
		    ,@(pattern-bind pattern targetval)
		    ,@procs)
	      (_match ,targetval ,@(cdr pattern-procs)))))))

(mac match (target & body)
  (_val pattern-procs (tuple (chunks body 2)))
  `(_match ,target ,@pattern-procs))

(mac def/match (fname & patterns)
  (_val argsym (gensym))
  `(def ,fname (& ,argsym)
  (match ,argsym
    ,@patterns)))

(mac defm (fname & patterns)
  (_val argsym (gensym))
  `(def ,fname (& ,argsym)
  (match ,argsym
    ,@patterns)))

(mac data (base-record-name & record-defs)
  `(do (record ,base-record-name ())
       ,@(map |quasiquote (record ,(get _ 0) ,base-record-name ,(get _ 1 None))| record-defs)))

(mac make-module (export & body)
  (def symbol->keyword (sym)
    (Keyword sym.name))
  (def make-exported-table (exported-symbols)
    (+ '(table) (tuple (flatten (map |make-tuple (symbol->keyword _) _| exported-symbols)))))
  (def make-exported-tuple (exported-symbols)
    (val tuple-name (gensym))
    (make-tuple
      `(record ,tuple-name Record ,exported-symbols)
      `(,tuple-name ,@exported-symbols)))
  `((fn () ,@body ,@(make-exported-tuple export))))

(mac module (name export & body)
  `(val ,name (make-module ,export ,@body)))

(mac del-hidden-vars ()
  (val hidden-var (gensym))
  `(for ,hidden-var (_get-hidden-vars)
    (if (in ,hidden-var (globals))
      (do
        (_del-hidden-var ,hidden-var)
        (delitem ,hidden-var (globals))))))

(mac del-hidden-vars-local ()
  (val hidden-var (gensym))
  `(for ,hidden-var (_get-hidden-vars)
    (if (in ,hidden-var (locals))
      (do
        (_del-hidden-var ,hidden-var)
        (delitem ,hidden-var (globals))))))


; itertools - recipes
(def take (n iterable)
  (tuple (islice iterable n)))

(def tabulate (func start)
  (map func (count start)))

(def quantify (iterable pred)
  (sum (map pred iterable)))

(def padnone (iterable)
  (chain iterable (repeat None)))

(def ncycles (iterable n)
  (chain.from_iterable (repeat (tuple iterable) n)))

(def dotproduct (t1 t2)
  (sum (map mul t1 t2)))

(def repeatfunc (func times & args)
  (if (is times None)
    (starmap func (repeat args)))
  (starmap func (repeat args times)))

(def pairwise (iterable)
  (val (a b) (tee iterable))
  (next b None)
  (zip a b))

(def grouper (iterable n fillvalue)
  (_val args (* (v (iter iterable)) n))
  (_val zip-longest (partial zip_longest :fillvalue fillvalue))
  (apply zip-longest args))

(def partition (pred iterable)
  (val (t1 t2) (tee iterable))
  (v (filterfalse pred t1) (filter pred t2)))

(def unique-justseen (iterable key)
  (map next (map (itemgetter 1) (groupby iterable key))))""")

    if not IS_PYPY:
        eval("""
(def powerset (iterable)
  (def combi (s)
    (for r (range (+ (len s) 1))
      (yield (combinations s r))))
  (chain.from_iterable (combi (tuple iterable))))

(def unique-everseen (iterable key)
  (_val seen (set))
  (_val seen-add seen.add)
  (if (is key None)
    (for element-a (filterfalse seen.__contains__ iterable)
      (seen-add element-a)
      (yield element-a))
    (for element-b iterable
      (_val k (key element-b))
      (if (not-in k seen)
        (do
          (seen-add k)
          (yield element-b))))))""")
    else:
        eval("""
(mac call/cl (callable)
  (val c (gensym))
  (val k (gensym))
  `(do
    (val ,c (continulet (fn (,k) (,callable (getattr ,k "switch")))))
    ((getattr ,c "switch"))))
        """)

    eval("""
(del-hidden-vars)
(del-hidden-vars-local)
;(val & &)
    """)

    for syntax in {'for', 'each', 'while', 'break', 'continue'}:
        del syntax_table[syntax]
        del global_env[syntax]
        del global_scope[syntax]

    if len(sys.argv) > 1:
        arg_parser = argparse.ArgumentParser(
            description='Mochi is a programming language.')
        arg_parser.add_argument('-v', '--version', action='version', version=VERSION)
        arg_parser.add_argument('-c', '--compile', action='store_true')
        arg_parser.add_argument('-e', '--execute-compiled-file', action='store_true')
        arg_parser.add_argument('file', nargs='?', type=str)
        args = arg_parser.parse_args()

        if args.file:
            if args.compile:
                output_code(compile_file(args.file, optimize=2))
            elif args.execute_compiled_file:
                execute_compiled_file(args.file)
            else:
                load_file(args.file, global_env)
            sys.exit(0)
    else:
        interact()


if __name__ == '__main__':
    main()
