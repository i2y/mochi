import ast

from .global_env import global_env
from mochi.parser.parser import Symbol


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
UNQUOTE_SPLICING = Symbol('unquote_splicing')
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
MAKE_TUPLE = Symbol('make_tuple')
MAKE_LIST = Symbol('make_list')
MAKE_DICT = Symbol('dict*')
WITH_DECORATOR = Symbol('with_decorator')
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


class Char(object):
    def __init__(self, str, lineno=0):
        self.value = str
        self.lineno = lineno

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
              'is_not': ast.IsNot(),
              'in': ast.In(),
              'not_in': ast.NotIn(),
              'and': ast.And(),
              'or': ast.Or()}