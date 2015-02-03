#!/usr/bin/env python
import sys
import traceback
from types import FunctionType
from functools import reduce, partial
from itertools import chain
from operator import concat, add
from collections import Iterator, Iterable, Sequence, MutableSequence
from io import StringIO

from pyrsistent import pmap, pclass

from .utils import issequence, issequence_except_str, is_tuple_or_list
from .constants import *
from .exceptions import UnquoteSplicingError, DuplicatedDefError, ReadError
from .global_env import global_env
from mochi.parser.parser import lex
from .translation import binding_name_set_stack, translator, Keyword, parse


def builtin(func):
    global_env[func.__name__] = func
    return func


def builtin_rename(new_name):
    def _builtin(func):
        global_env[new_name] = func
        return func

    return _builtin


#@builtin_rename('uniq')
#@builtin_rename('gensym')


@builtin
def flip(func):
    return lambda arg0, arg1, *rest: func(arg1, arg0, *rest)


@builtin
def bind(next_func, input):
    if input is None:
        return None
    return next_func(input)


@builtin
def mapchain(func, iterable):
    return reduce(chain, map(func, iterable))


@builtin
def mapcat(func, iterable):
    return reduce(concat, map(func, iterable))


@builtin
def mapadd(func, iterable):
    return reduce(add, map(func, iterable))


@builtin_rename('on_err')
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


@builtin
def gen_with(context, func):
    with context:
        retval = func(context)
        if isinstance(retval, Iterable):
            # yield from retval
            for value in retval:
                yield value
        else:
            yield retval


#@builtin_rename('with')
#def _with(context, func):
#    with context:
#        return func(context)


@builtin
def gen(func, iterable):
    while True:
        yield func(next(iterable))


@builtin
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


class SexpReader(object):
    def __init__(self, port):
        object.__init__(self)
        self.port = port
        self.nest_level = 0
        self.line = 1
        self.commands = ("'", '(', ',', '`', '"', ';', '@', '~', '[', '{')
        self.white_spaces = (' ', '\r', '\n', '\t')
        self.separations = self.commands + self.white_spaces + (')', '~', ']', '}', EOF, '#')
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
        elif self.port.next_char() == '~':
            rest_exps = self.read_delimited_list('~')
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
current_error_port = OutputPort(file=sys.__stderr__)


@builtin
def display(obj):
    return current_output_port.display(obj)


@builtin
def write(obj, end='\n'):
    if obj is not None:
        current_output_port.write(obj)
        current_output_port.write(end)


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
    return eval(''.join(l), {'func': func})


@builtin
def rcurried(func):
    argcount = func.__code__.co_argcount
    l = ['lambda arg_{0}: '.format(n) for n in range(argcount - 1, -1, -1)]
    l.append('func(')
    l.extend(['arg_{0}, '.format(n) for n in range(argcount - 1)])
    l.extend(['arg_', str(argcount - 1), ')'])
    return eval(''.join(l), {'func': func})


@builtin
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


@builtin
def unquote_splice(lis):
    newlis = []
    splicing_flag = False
    for item in lis:
        if type(item) is Symbol and item.name == '_mochi_unquote_splicing':
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


def eval_tokens(tokens):
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
                # print(file=current_output_port)
    except Exception:
        traceback.print_exc(file=current_error_port)


@builtin_rename('eval')
def eval_code_block(block):
    lexer = lex(block + '\n', repl_mode=True)
    eval_tokens(lexer)


def eval_sexp_str(sexp_str):
    sr = SexpReader(InputPort(StringIO(sexp_str)))
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


@builtin_rename('_del_hidden_var')
def del_hidden_var(name):
    translator.hidden_vars.remove(name)


@builtin_rename('_get_hidden_vars')
def get_hidden_vars():
    return translator.hidden_vars


@builtin_rename('delitem')
def _del(name, env):
    del env[name]


@builtin_rename('macex1')
@builtin_rename('macroexpand_1')
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


@builtin
def is_macro(func):
    return isinstance(func, FunctionType) and translator.has_macro(func.__name__)


@builtin
def is_function(func):
    return isinstance(func, FunctionType) and (not translator.has_macro(func.__name__))


@builtin
def iter_except(func, exc):
    try:
        while 1:
            yield func()
    except exc:
        pass


@builtin_rename('assert')
def _assert(value):
    assert value


@builtin
def assert_equal(value1, value2):
    assert value1 == value2


@builtin
def assert_not_equal(value1, value2):
    assert value1 != value2


@builtin
def assert_none(value):
    assert value is None


@builtin
def assert_not_none(value):
    assert value is not None


@builtin
def assert_same(value1, value2):
    assert value1 is value2


@builtin
def assert_not_same(value1, value2):
    assert value1 is not value2
