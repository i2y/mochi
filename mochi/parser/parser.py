from queue import Queue
import warnings
from collections import Sequence

from rply import ParserGenerator, LexerGenerator, Token, ParsingError


class Symbol(object):
    def __init__(self, name, lineno=0, col_offset=0):
        self.name = name
        self.outer_name = name
        self.lineno = lineno
        self.col_offset = col_offset

    def eval(self, env):
        pass

    def __repr__(self):
        return self.outer_name

    def __str__(self):
        return self.outer_name

    def __eq__(self, other):
        if type(other) is not Symbol:
            return False
        if self.name == other.name:
            return True
        else:
            return False

    def __hash__(self):
        return (self.name.__hash__() << 16) + self.outer_name.__hash__()


class Keyword(object):
    def __init__(self, name, lineno=0, col_offset=0):
        self.name = name
        self.lineno = lineno
        self.col_offset = col_offset
        self.repr = ':' + self.name

    def __repr__(self):
        return self.repr

    def __str__(self):
        return self.name

    def __call__(self, table):
        return table[self.name]

    def __eq__(self, other):
        if type(other) is not Keyword:
            return False
        if self.name == other.name:
            return True
        else:
            return False

    def __hash__(self):
        return self.name.__hash__()


lg = LexerGenerator()

lg.add('TQUOTE_STR', r'(?x)"""(?:|[^\\]|\\.|\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8})*"""')
lg.add('SQUOTE_STR', r"(?x)'(?:|[^'\\]|\\.|\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8})*'")
lg.add('DQUOTE_STR', r'(?x)"(?:|[^"\\]|\\.|\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8})*"')

lg.add('TQUOTE_RAW_STR', r'(?x)r"""(?:|[^\\]|\\.|\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8})*"""')
lg.add('SQUOTE_RAW_STR', r"(?x)r'(?:|[^'\\]|\\.|\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8})*'")
lg.add('DQUOTE_RAW_STR', r'(?x)r"(?:|[^"\\]|\\.|\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8})*"')

lg.add('NUMBER', r'-?[0-9]+(?:\.[0-9]+)?')
lg.add('DOT_NAME', r'\.\&?[_a-zA-Z$][-_a-zA-Z0-9]*')
lg.add('NAME', r'\&?[_a-zA-Z$][-_a-zA-Z0-9]*')
lg.add('PIPELINE_FIRST_BIND', r'\|>1\?')
lg.add('PIPELINE_FIRST', r'\|>1')
lg.add('PIPELINE_BIND', r'\|>\?')
lg.add('PIPELINE', r'\|>')
lg.add('PIPELINE_SEND', r'!>')
lg.add('PIPELINE_MULTI_SEND', r'!&>')
lg.add('BAR', r'\|')
lg.add('LBRACK', r'\[')
lg.add('RBRACK', r'\]')
lg.add('LBRACE', r'\{')
lg.add('RBRACE', r'\}')
lg.add('LPAREN', r'\(')
lg.add('RPAREN', r'\)')
lg.add('PERCENT', r'%')
lg.add('COMMA', r',')
lg.add('THINARROW', r'->')
lg.add('COLON', r':')
lg.add('CALET', r'\^')
lg.add('OPPLUS', r'\+')
lg.add('OPMINUS', r'-')
lg.add('OPTIMES', r'\*')
lg.add('OPDIV', r'/')
lg.add('OPLEQ', r'<=')
lg.add('OPGEQ', r'>=')
lg.add('OPEQ', r'==')
lg.add('OPNEQ', r'!=')
lg.add('OPLT', r'<')
lg.add('OPGT', r'>')
lg.add('BANG', r'!')

lg.add('EQUALS', r'=')
lg.add('SEMI', r';')
lg.add('AT', r'@')
lg.add('AMP', r'\&')
lg.add('BACKSLASH', r'\\')

lg.add('NEWLINE', r'(?:(?:\r?\n)[\t ]*)+')
lg.ignore(r'[ \t\f\v]+')
lg.ignore(r'#.*(?:\n|\r|\r\n|\n\r|$)')  # comment

klg = LexerGenerator()
klg.add('IMPORT', r'^import$')
klg.add('MODULE', r'^module$')
klg.add('REQUIRE', r'^require$')
klg.add('EXPORT', r'^export$')
klg.add('VAR', r'^var$')
klg.add('LET', r'^let$')
klg.add('DEF', r'^def$')
klg.add('DEFM', r'^defm$')
klg.add('FN', r'^fn$')
klg.add('TRUE', r'^True$')
klg.add('FALSE', r'^False$')
klg.add('DOC', r'^doc:$')
klg.add('TRY', r'^try$')
klg.add('EXCEPT', r'^except$')
klg.add('AS', r'^as$')
klg.add('FINALLY', r'^finally$')
klg.add('RAISE', r'^raise$')
klg.add('IF', r'^if$')
klg.add('ELSE', r'^else$')
klg.add('ELSEIF', r'^elif$')
klg.add('MATCH', r'^match$')
klg.add('RECEIVE', r'^receive$')
klg.add('OF', r'^of$')
klg.add('RECORD', r'^record$')
klg.add('DATA', r'^data$')
klg.add('YIELD', r'^yield$')
klg.add('RETURN', r'^return$')
klg.add('WITH', r'^with$')
klg.add('MACRO', r'^macro$')
klg.add('QUOTE', r'^quote$')
klg.add('QUASI_QUOTE', r'^quasi_quote$')
klg.add('UNQUOTE', r'^unquote$')
klg.add('UNQUOTE_SPLICING', r'^unquote_splicing$')
klg.add('FOR', r'^for$')
klg.add('IN', r'^in$')
klg.add('FROM', r'^from$')
klg.add('OPAND', r'^and$')
klg.add('OPOR', r'^or$')
klg.add('OPIS', r'^is$')
klg.add('NOT', r'^not$')

pg = ParserGenerator(['NUMBER', 'OPPLUS', 'OPMINUS', 'OPTIMES', 'OPDIV', 'OPLEQ', 'OPGEQ', 'OPEQ', 'OPNEQ',
                      'OPLT', 'OPGT', 'OPAND', 'OPOR', 'OPIS', 'NOT', 'NEWLINE', 'PERCENT', 'EXPORT',
                      'LPAREN', 'RPAREN', 'TRUE', 'FALSE', 'TQUOTE_STR', 'DQUOTE_STR', 'SQUOTE_STR',
                      'AT', 'BANG', 'DOT_NAME', 'TQUOTE_RAW_STR', 'DQUOTE_RAW_STR', 'SQUOTE_RAW_STR',
                      'NAME', 'EQUALS', 'IF', 'ELSEIF', 'ELSE', 'COLON', 'SEMI', 'DATA', 'IMPORT', 'REQUIRE',
                      'LBRACK', 'RBRACK', 'COMMA', 'DEF', 'DOC', 'CALET', 'PIPELINE', 'PIPELINE_BIND', 'PIPELINE_FIRST',
                      'PIPELINE_FIRST_BIND', 'PIPELINE_SEND', 'PIPELINE_MULTI_SEND', 'RETURN',
                      'LBRACE', 'RBRACE', 'MATCH', 'DEFM', 'RECORD', 'AMP', 'FN', 'THINARROW', 'RECEIVE',
                      'YIELD', 'FROM', 'FOR', 'IN', 'INDENT', 'DEDENT', 'TRY', 'FINALLY', 'EXCEPT',
                      'MODULE', 'AS', 'RAISE', 'WITH', 'MACRO', 'QUOTE', 'QUASI_QUOTE', 'UNQUOTE', 'UNQUOTE_SPLICING'],
                     precedence=[('left', ['EQUALS']),
                                 ('left', ['NOT']),
                                 ('left', ['OPIS']),
                                 ('left', ['OPEQ', 'OPLEQ', 'OPGEQ', 'OPNEQ', 'OPLT', 'OPGT', 'OPAND', 'OPOR',
                                           'PIPELINE', 'PIPELINE_BIND', 'PIPELINE_FIRST', 'PIPELINE_FIRST_BIND',
                                           'PIPELINE_SEND', 'PIPELINE_MULTI_SEND']),
                                 ('left', ['OPPLUS', 'OPMINUS']),
                                 ('left', ['LBRACK', 'RBRACK']),
                                 ('left', ['OPTIMES', 'OPDIV', 'PERCENT'])],
                     cache_id='mochi')


@pg.production('program : block')
def program(p):
    return p[0]


@pg.production('block : stmts')
def block(p):
    return p[0]


@pg.production('stmts : stmts stmt')
def stmts_b(p):
    if p[1] is None:
        return p[0]
    else:
        return p[0] + [p[1]]


@pg.production('stmts : stmt')
def stmts_stmt(p):
    if p[0] is None:
        return []
    else:
        return [p[0]]


@pg.production('stmt : NEWLINE')
@pg.production('stmt : SEMI')
def stmt_newline(p):
    pass


@pg.production('stmt : binop_expr')
@pg.production('stmt : let_expr')
@pg.production('stmt : deco_expr')
@pg.production('stmt : def_expr')
@pg.production('stmt : defm_expr')
@pg.production('stmt : record_expr')
@pg.production('stmt : data_expr')
@pg.production('stmt : import_expr')
@pg.production('stmt : require_expr')
@pg.production('stmt : module_expr')
@pg.production('stmt : from_expr')
@pg.production('stmt : if_expr')
@pg.production('stmt : try_stmt')
@pg.production('stmt : with_stmt')
@pg.production('stmt : raise_stmt')
@pg.production('stmt : return_stmt')
@pg.production('stmt : macro_stmt')
@pg.production('stmt : q_stmt')
@pg.production('stmt : qq_stmt')
def stmt(p):
    return p[0]


@pg.production('import_expr : IMPORT names_list')
def import_expr(p):
    return [Symbol('import')] + p[1]


@pg.production('names_list : names_list COMMA names')
def names(p):
    return p[0] + [p[2]]


@pg.production('names_list : names')
def names_single(p):
    return [p[0]]


@pg.production('names : _names')
def names(p):
    return Symbol('.'.join(p[0]))


@pg.production('_names : NAME')
def _names_one(p):
    return [p[0].getstr()]


@pg.production('_names : _names DOT_NAME')
def _names(p):
    return p[0] + [p[1].getstr()[1:]]


@pg.production('require_expr : REQUIRE string')
def require_expr(p):
    return [Symbol('require'), p[1]]


@pg.production('module_expr : MODULE NAME COLON NEWLINE INDENT export_cls_list block DEDENT')
def module_expr(p):
    return [Symbol('module'), token_to_symbol(p[1]), p[5]] + p[6]


@pg.production('export_cls_list : export_cls_list NEWLINE export_cls')
def export_cls_list(p):
    return p[0] + p[2]


@pg.production('export_cls_list : export_cls')
def export_cls_list_one(p):
    return p[0]


@pg.production('export_cls : EXPORT namelist')
def export_cls(p):
    return p[1]


@pg.production('namelist : namelist COMMA name')
def names(p):
    return p[0] + [p[2]]


@pg.production('namelist : name')
def names_single(p):
    return [p[0]]


@pg.production('name : NAME')
def name(p):
    return token_to_symbol(p[0])


@pg.production('tuple_elt : binop_expr COMMA')
def tuple_elt(p):
    return p[0]


@pg.production('from_expr : FROM names IMPORT namelist')
def from_expr(p):
    return [Symbol('from_import'), p[1], p[3]]


@pg.production('suite : binop_expr')  # TODO multi
def suite_expr(p):
    return p[0]


@pg.production('suite : NEWLINE INDENT stmts DEDENT')
def suite_stmts(p):
    return [Symbol('do')] + p[2]


@pg.production('suite2 : NEWLINE INDENT stmts DEDENT')
def suite2_stmts(p):
    return p[2]


@pg.production('try_stmt : TRY COLON suite2 finally_cls')
def try_finally_stmt(p):
    return [Symbol('try')] + p[2] + [p[3]]


@pg.production('try_stmt : TRY COLON suite2 except_cls_list')
def try_except_stmt(p):
    return [Symbol('try')] + p[2] + p[3]


@pg.production('try_stmt : TRY COLON suite2 except_cls_list finally_cls')
def try_excepts_finally_stmt(p):
    return [Symbol('try')] + p[2] + p[3] + [p[4]]


@pg.production('except_cls_list : except_cls_list except_cls')
def except_cls_list(p):
    return p[0] + [p[1]]


@pg.production('except_cls_list : except_cls')
def except_cls_list(p):
    return [p[0]]


@pg.production('except_cls : EXCEPT binop_expr AS NAME COLON suite2')
def except_cls(p):
    return [Symbol('except'), p[1], token_to_symbol(p[3])] + p[5]


@pg.production('finally_cls : FINALLY COLON suite2')
def finally_cls(p):
    return [Symbol('finally')] + p[2]


@pg.production('raise_stmt : RAISE binop_expr')
def raise_stmt(p):
    return [Symbol('raise'), p[1]]


@pg.production('macro_stmt : MACRO fun_header COLON suite2')
def macro_stmt(p):
    fun_name, fun_args = p[1]
    return [Symbol('mac'), fun_name, fun_args] + p[3]


@pg.production('q_stmt : QUOTE COLON suite')
def q_stmt(p):
    return [Symbol('quote'), p[2]]


@pg.production('quote_expr : QUOTE LPAREN binop_expr RPAREN')
def quote_expr(p):
    return [Symbol('quote'), p[2]]


@pg.production('qq_stmt : QUASI_QUOTE COLON suite')
def qq_stmt(p):
    return [Symbol('quasiquote'), p[2]]


@pg.production('quasi_quote_expr : QUASI_QUOTE LPAREN binop_expr RPAREN')
def quasi_quote_expr(p):
    return [Symbol('quasiquote'), p[2]]


@pg.production('uq_expr : UNQUOTE LPAREN binop_expr RPAREN')
def qq_stmt(p):
    return [Symbol('unquote'), p[2]]


@pg.production('uqs_expr : UNQUOTE_SPLICING LPAREN binop_expr RPAREN')
def qq_stmt(p):
    return [Symbol('unquote_splicing'), p[2]]


@pg.production('with_stmt : WITH with_contexts COLON suite2')
def with_stmt(p):
    return [Symbol('with'), p[1]] + p[3]


@pg.production('with_contexts : with_contexts COMMA with_context')
def with_contexts(p):
    return p[0] + [p[2]]


@pg.production('with_contexts : with_context')
def with_contexts_one(p):
    return [p[0]]


@pg.production('with_context : binop_expr AS NAME')
def with_context(p):
    return [p[0], token_to_symbol(p[2])]


@pg.production('return_stmt : RETURN binop_expr')
def raise_stmt(p):
    return [Symbol('return'), p[1]]


def token_to_symbol(token):
    return Symbol(token.getstr(), token.getsourcepos().lineno, token.getsourcepos().colno)


def token_to_keyword(token):
    return Keyword(token.getstr(), token.getsourcepos().lineno, token.getsourcepos().colno)


@pg.production('let_expr : binop_expr EQUALS binop_expr')
def let_expr(p):
    return [Symbol('val', 0, 0), p[0], p[2]]


@pg.production('binding : NAME')
def binding(p):
    return token_to_symbol(p[0])


@pg.production('expr : fn_expr')
@pg.production('expr : paren_expr')
# @pg.production('expr : if_expr')
@pg.production('expr : trailing_if_expr')
@pg.production('expr : prim_expr')
@pg.production('expr : uq_expr')
@pg.production('expr : uqs_expr')
@pg.production('expr : app_expr')
@pg.production('expr : left_app_expr')
@pg.production('expr : dict_expr')
@pg.production('expr : tuple_expr')
@pg.production('expr : match_expr')
@pg.production('expr : receive_expr')
@pg.production('expr : yield_expr')
@pg.production('expr : yield_from_expr')
@pg.production('expr : for_expr')
@pg.production('expr : block_expr')
@pg.production('expr : dot_expr')
@pg.production('expr : get_expr')
@pg.production('expr : send_msg_expr')
@pg.production('expr : quote_expr')
@pg.production('expr : quasi_quote_expr')
@pg.production('expr : id_expr')
#@pg.production('expr : app_nc_expr')
def expr(p):
    return p[0]


@pg.production('paren_expr : LPAREN binop_expr RPAREN')
def paren_expr(p):
    return p[1]


@pg.production('prim_expr : NUMBER')
def expr_num(p):
    num_repr = p[0].getstr()
    try:
        return int(num_repr)
    except ValueError as _:
        return float(num_repr)


@pg.production('prim_expr : string')
def expr_string(p):
    return p[0]


@pg.production('string : DQUOTE_STR')
@pg.production('string : SQUOTE_STR')
def expr_quote_str(p):
    return quote_str(p[0].getstr()[1:-1])


@pg.production('string : TQUOTE_STR')
def expr_triple_quote_str(p):
    return quote_str(p[0].getstr()[3:-3])


def quote_str(string):
    new_string = ''
    string_enumerator = enumerate(string)
    for index, char in string_enumerator:
        if char == '\\':
            index, char = next(string_enumerator)
            if char == 'n':
                char = '\n'
            elif char == 't':
                char = '\t'
            elif char == 'r':
                char = '\r'
            elif char in {'\\', "'", '"'}:
                pass
            else:
                char = '\\' + char
        new_string = new_string + char
    return new_string


@pg.production('string : DQUOTE_RAW_STR')
@pg.production('string : SQUOTE_RAW_STR')
def expr_quote_raw_str(p):
    return p[0].getstr()[2:-1]


@pg.production('string : TQUOTE_RAW_STR')
def expr_triple_quote_raw_str(p):
    return p[0].getstr()[4:-3]


@pg.production('prim_expr : bool_expr')
def expr_false(p):
    return p[0]


@pg.production('bool_expr : TRUE')
def expr_true(p):
    return Symbol('True')


@pg.production('bool_expr : FALSE')
def expr_false(p):
    return Symbol('False')


@pg.production('id_expr : NAME')
def id_expr(p):
    return token_to_symbol(p[0])


@pg.production('id_expr : AMP')
def id_expr(p):
    return Symbol('&')


@pg.production('if_expr : IF binop_expr COLON suite elseif_exprs')
def if_expr(p):
    if p[4] is None:
        return [Symbol('if'), p[1], p[3]]
    else:
        return [Symbol('if'), p[1], p[3]] + p[4]


@pg.production('if_expr : IF binop_expr COLON suite elseif_exprs ELSE COLON suite')
def if_else_expr(p):
    if p[4] is None:
        return [Symbol('if'), p[1], p[3], p[7]]
    else:
        return [Symbol('if'), p[1], p[3]] + p[4] + [p[7]]


@pg.production('elseif_exprs : elseif_exprs elseif_expr')
def elseif_exprs(p):
    return p[0] + p[1]


@pg.production('elseif_exprs : elseif_expr')
def elseif_exprs_expr(p):
    return p[0]


@pg.production('elseif_expr : ELSEIF binop_expr COLON suite')
def elseif_expr(p):
    return [p[1], p[3]]


@pg.production('elseif_expr :')
def elseif_expr_empty(p):
    return None


@pg.production('trailing_if_expr : binop_expr IF binop_expr ELSE binop_expr')
def trailing_if_expr(p):
    return [Symbol('if'), p[2], p[0], p[4]]


@pg.production('yield_expr : YIELD binop_expr')
def yield_expr(p):
    return [Symbol('yield'), p[1]]


@pg.production('yield_from_expr : YIELD FROM binop_expr')
def yield_from_expr(p):
    return [Symbol('yield_from'), p[1]]


def issequence(obj):
    return isinstance(obj, Sequence)


def issequence_except_str(obj):
    if isinstance(obj, str):
        return False
    return isinstance(obj, Sequence)


def _compute_underscore_max_num(exps):
    max_num = 0

    if not issequence_except_str(exps):
        exps = (exps,)

    for exp in exps:
        if isinstance(exp, Symbol) and exp.name.startswith('$'):
            try:
                n = int(exp.name[1:])
            except:
                n = 1
        elif issequence_except_str(exp):
            n = _compute_underscore_max_num(exp)
        else:
            n = 0

        if n > max_num:
            max_num = n
    return max_num


@pg.production('dot_expr : expr DOT_NAME')
def dot_expr(p):
    return [Symbol('getattr'), p[0], p[1].getstr()[1:]]


@pg.production('get_expr : binop_expr LBRACK binop_expr RBRACK')
def get_expr(p):
    return [Symbol('get'), p[0], p[2]]


@pg.production('get_expr : binop_expr LBRACK range_start COLON range_end RBRACK')
def get_slice_expr(p):
    return [Symbol('get'), p[0], p[2], p[4]]


@pg.production('get_expr : binop_expr LBRACK range_start COLON range_end COLON range_interval RBRACK')
def get_slice_expr(p):
    return [Symbol('get'), p[0], p[2], p[4], p[6]]


@pg.production('range_start : ')
@pg.production('range_end : ')
@pg.production('range_interval : ')
def range_start_none(p):
    return Symbol('None')


@pg.production('range_start : binop_expr')
@pg.production('range_end : binop_expr')
@pg.production('range_interval : binop_expr')
def range_start_none(p):
    return p[0]


@pg.production('send_msg_expr : expr BANG expr')
def dot_expr(p):
    return [Symbol('send'), p[2], p[0]]


@pg.production('for_expr : LBRACK binop_expr FOR pattern IN binop_expr RBRACK')
def for_expr(p):
    pattern = p[3]
    items = p[5]
    body = p[1]
    return [Symbol('tuple_of'), body, [pattern, items]]


@pg.production('for_expr : LBRACK binop_expr FOR pattern IN binop_expr IF binop_expr RBRACK')
def for_expr_if(p):
    pattern = p[3]
    items = p[5]
    body = p[1]
    when = p[7]
    return [Symbol('tuple_of'), body, [pattern, items, Keyword('when'), when]]


@pg.production('tuple_expr : LBRACK tuple_elts binop_expr RBRACK')
def tuple_expr(p):
    return [Symbol('make_tuple')] + p[1] + [p[2]]


@pg.production('tuple_expr : LBRACK binop_expr RBRACK')
def tuple_expr_one(p):
    return [Symbol('make_tuple'), p[1]]


@pg.production('tuple_expr : LBRACK tuple_elts binop_expr RBRACK')
def tuple_expr(p):
    return [Symbol('make_tuple')] + p[1] + [p[2]]


@pg.production('tuple_expr : LBRACK binop_expr RBRACK')
def tuple_expr_one(p):
    return [Symbol('make_tuple'), p[1]]


@pg.production('tuple_expr : LBRACK RBRACK')
def tuple_expr_empty(p):
    return [Symbol('make_tuple')]


@pg.production('tuple_elts : tuple_elts tuple_elt')
def tuple_elts(p):
    return p[0] + [p[1]]


@pg.production('tuple_elts : tuple_elt')
def tuple_elts_elt(p):
    return [p[0]]


@pg.production('tuple_elt : binop_expr COMMA')
def tuple_elt(p):
    return p[0]


@pg.production('deco_expr : decorators def_expr')
def deco_expr(p):
    # return p[1][:2] + p[0] + p[1][2:]
    return [Symbol('with_decorator')] + p[0] + [p[1]]


@pg.production('decorators : decorators decorator')
def decorators(p):
    return p[0] + [p[1]]


@pg.production('decorators : decorator')
def decorators_single(p):
    return [p[0]]


@pg.production('decorator : AT binop_expr NEWLINE')
def decorator(p):
    return p[1]


@pg.production('def_expr : DEF fun_header doc_string COLON suite')
def fun_expr(p):
    fun_name, fun_args = p[1]
    return [Symbol('def'), fun_name, fun_args, p[4]]


@pg.production('defm_expr : DEF NAME doc_string COLON NEWLINE INDENT defm_case_branches DEDENT')
def fun_expr(p):
    return [Symbol('defm'), token_to_symbol(p[1])] + p[6]


@pg.production('defm_case_branches : defm_case_branches defm_case_branch')
def case_branches(p):
    return p[0] + p[1]


@pg.production('defm_case_branches : defm_case_branch')
def case_branches_branch(p):
    return p[0]


@pg.production('defm_case_branch : defm_pattern COLON NEWLINE INDENT stmts DEDENT')
def case_branch(p):
    return [p[0], [Symbol('do')] + p[4]]


@pg.production('defm_case_branch : defm_pattern COLON binop_expr NEWLINE')
def case_branch(p):
    return [p[0], p[2]]


@pg.production('defm_case_branch : defm_pattern COLON binop_expr SEMI')
def case_branch(p):
    return [p[0], p[2]]


@pg.production('defm_pattern : app_nc_args')
def pattern(p):
    return p[0]


@pg.production('defm_pattern : pattern')
def app_args(p):
    return [p[0]]


@pg.production('defm_pattern : pattern COMMA defm_pattern')
def app_args(p):
    return [p[0]] + p[2]


@pg.production('fun_header : NAME args')
def fun_header(p):
    return [token_to_symbol(p[0]), p[1]]


@pg.production('fn_expr : args THINARROW suite')
def fun_expr(p):
    return [Symbol('fn'), p[0], p[2]]


@pg.production('args : LPAREN list_arg_elts id_expr RPAREN')
def args(p):
    return p[1] + [p[2]]


@pg.production('args : LPAREN id_expr RPAREN')
def args_one(p):
    return [p[1]]


@pg.production('args : LPAREN RPAREN')
def args_empty(p):
    return []


@pg.production('nc_args : list_arg_elts id_expr')
def args(p):
    return p[0] + [p[1]]


@pg.production('nc_args : id_expr')
def args_one(p):
    return [p[0]]


@pg.production('list_arg_elts : list_arg_elts list_arg_elt')
def list_arg_elts(p):
    return p[0] + [p[1]]


@pg.production('list_arg_elts : list_arg_elt')
def list_arg_elts_elt(p):
    return [p[0]]


@pg.production('list_arg_elt : id_expr COMMA')
def list_arg_elt(p):
    return p[0]


def _create_underscore_args(exps):
    max_num = _compute_underscore_max_num(exps)
    if max_num == 1:
        return [Symbol('$1')]
    else:
        return [Symbol('$' + str(n)) for n in range(1, max_num + 1)]


@pg.production('block_expr : THINARROW suite')
def block_expr(p):
    block = p[1]
    return [Symbol('fn'), _create_underscore_args(block), block]


@pg.production('doc_string : DOC string')
@pg.production('doc_string : ')
def doc_string(p):
    pass


@pg.production('app_expr : expr app_args')
def app_expr(p):
    return [p[0]] + p[1]


@pg.production('app_args : LPAREN app_args_elts RPAREN')
def app_args(p):
    return p[1]


@pg.production('app_args : LPAREN RPAREN')
def app_args(p):
    return []


@pg.production('app_args_elts : app_args_elts COMMA app_args_elt')
def app_args_elts(p):
    return p[0] + p[2]


@pg.production('app_args_elts : app_args_elt')
def app_args_elts(p):
    return p[0]


@pg.production('app_args_elt : NAME EQUALS binop_expr')
def app_args_elt(p):
    return [token_to_keyword(p[0]), p[2]]


@pg.production('app_args_elt : EQUALS NAME')
def app_args_elt_short(p):
    return [token_to_keyword(p[1]), token_to_symbol(p[1])]


@pg.production('app_args_elt : binop_expr')
def app_args_elt(p):
    return [p[0]]


@pg.production('app_expr : expr app_args fn_expr')
@pg.production('app_expr : expr app_args block_expr')
def trailing_closure_expr(p):
    return [p[0], p[2]] + p[1]


@pg.production('app_expr : expr app_args app_args')
@pg.production('app_expr : expr app_args app_args')
def trailing_closure_expr(p):
    return [[p[0]] + p[1]] + p[2]


@pg.production('app_expr : expr app_args AT fn_expr')
@pg.production('app_expr : expr app_args AT block_expr')
def trailing_closure_expr(p):
    return [p[0]] + p[1] + [p[3]]


@pg.production('app_nc_expr : expr app_nc_args')
def app_expr(p):
    return [p[0]] + p[1]


@pg.production('app_nc_args : app_nc_arg')
def app_args(p):
    return [p[0]]


@pg.production('app_nc_args : app_nc_arg COMMA app_nc_args')
def app_args(p):
    return [p[0]] + p[2]


@pg.production('app_nc_arg : binop_expr')
def app_nc_arg(p):
    return p[0]


@pg.production('left_app_expr : expr CALET left_app_fun_expr app_args')
def left_app_expr(p):
    expr, _, left_app_fun_expr, app_args = p
    return [left_app_fun_expr, expr] + app_args


@pg.production('left_app_fun_expr : id_expr')
def left_app_fun_expr(p):
    return p[0]


@pg.production('dict_expr : LBRACE RBRACE')
def dict_expr_empty(p):
    return [Symbol('table')]


@pg.production('dict_expr : LBRACE fields RBRACE')
def dict_expr(p):
    return [Symbol('table')] + p[1]


@pg.production('fields : field')
def fields_one(p):
    return p[0]


@pg.production('fields : list_fields field')
def fields(p):
    return p[0] + p[1]


@pg.production('list_fields : list_field')
def list_fields_one(p):
    return p[0]


@pg.production('list_fields : list_fields list_field')
def list_fields(p):
    return p[0] + p[1]


@pg.production('list_field : field COMMA')
def list_field(p):
    return p[0]


@pg.production('field : key COLON binop_expr')
def field(p):
    return [p[0], p[2]]


@pg.production('field : EQUALS NAME')
def field(p):
    s = token_to_symbol(p[1])
    return [s.name, s]


@pg.production('key : binop_expr')
def key(p):
    return p[0]


@pg.production('match_expr : MATCH expr COLON NEWLINE INDENT case_branches DEDENT')
def case(p):
    return [Symbol('match'), p[1]] + p[5]


@pg.production('case_branches : case_branches case_branch')
def case_branches(p):
    return p[0] + p[1]


@pg.production('case_branches : case_branch')
def case_branches_branch(p):
    return p[0]


@pg.production('case_branch : pattern COLON NEWLINE INDENT stmts DEDENT')
def case_branch(p):
    return [p[0], [Symbol('do')] + p[4]]


@pg.production('case_branch : pattern COLON binop_expr NEWLINE')
def case_branch(p):
    return [p[0], p[2]]


@pg.production('case_branch : pattern COLON binop_expr SEMI')
def case_branch(p):
    return [p[0], p[2]]


# @pg.production('pattern : fn_expr')
@pg.production('pattern : prim_pattern')
@pg.production('pattern : dict_pattern')
@pg.production('pattern : sequence_pattern')
@pg.production('pattern : sequence_type_pattern')
@pg.production('pattern : type_pattern')
@pg.production('pattern : id_pattern')
@pg.production('pattern : and_pattern')
@pg.production('pattern : or_pattern')
@pg.production('pattern : quote_pattern')
# TODO @pg.production('defm_pattern : app_nc_args')
def pattern(p):
    return p[0]


@pg.production('prim_pattern : NUMBER')
def pattern_num(p):
    num_repr = p[0].getstr()
    try:
        return int(num_repr)
    except ValueError as _:
        return float(num_repr)


@pg.production('prim_pattern : string')
def pattern_string(p):
    return p[0]


@pg.production('prim_pattern : bool_expr')
def pattern_bool(p):
    return p[0]


@pg.production('dict_pattern : LBRACE RBRACE')
def dict_pattern_empty(p):
    return [Symbol('table')]


@pg.production('dict_pattern : LBRACE dict_pattern_fields RBRACE')
def dict_pattern(p):
    return [Symbol('table')] + p[1]


@pg.production('dict_pattern_fields : dict_pattern_field')
def fields_one(p):
    return p[0]


@pg.production('dict_pattern_fields : dict_pattern_list_fields dict_pattern_field')
def fields(p):
    return p[0] + p[1]


@pg.production('dict_pattern_list_fields : dict_pattern_list_field')
def list_fields_one(p):
    return p[0]


@pg.production('dict_pattern_list_fields : dict_pattern_list_fields dict_pattern_list_field')
def list_fields(p):
    return p[0] + p[1]


@pg.production('dict_pattern_list_field : dict_pattern_field COMMA')
def list_field(p):
    return p[0]


@pg.production('dict_pattern_field : dict_pattern_key COLON pattern')
def field(p):
    return [p[0], p[2]]


@pg.production('dict_pattern_field : EQUALS NAME')
def field(p):
    s = token_to_symbol(p[1])
    return [s.name, s]


@pg.production('dict_pattern_key : binop_expr')
def key(p):
    return p[0]


@pg.production('id_pattern : NAME')
def id_pattern(p):
    return token_to_symbol(p[0])


@pg.production('id_pattern : AMP')
def id_pattern(p):
    return Symbol('&')


@pg.production('sequence_pattern : LBRACK sequence_pattern_elts pattern RBRACK')
def sequence_pattern(p):
    return [Symbol('make_tuple')] + p[1] + [p[2]]


@pg.production('sequence_pattern : LBRACK pattern RBRACK')
def sequence_pattern_one(p):
    return [Symbol('make_tuple'), p[1]]


@pg.production('sequence_pattern : LBRACK RBRACK')
def sequence_pattern_empty(p):
    return [Symbol('make_tuple')]


@pg.production('sequence_pattern_elts : sequence_pattern_elts sequence_pattern_elt')
def sequence_pattern_elts(p):
    return p[0] + [p[1]]


@pg.production('sequence_pattern_elts : sequence_pattern_elt')
def sequence_pattern_elts_elt(p):
    return [p[0]]


@pg.production('sequence_pattern_elt : pattern COMMA')
def sequence_pattern_elt(p):
    return p[0]


@pg.production('sequence_type_pattern : names LPAREN sequence_pattern_elts pattern RPAREN')
def sequence_type_pattern(p):
    return [Symbol('sequence_type'), p[0]] + p[2] + [p[3]]


@pg.production('sequence_type_pattern : names LPAREN pattern RPAREN')
def sequence_type_pattern_one(p):
    return [Symbol('sequence_type'), p[0], p[2]]


@pg.production('and_pattern : pattern OPAND pattern')
def and_pattern(p):
    return [token_to_symbol(p[1]), p[0], p[2]]


@pg.production('or_pattern : pattern OPOR pattern')
def or_pattern(p):
    return [token_to_symbol(p[1]), p[0], p[2]]


@pg.production('type_pattern : NAME pattern')
def type_pattern(p):
    return [Symbol('type'), token_to_symbol(p[0]), p[1]]


@pg.production('quote_pattern : QUOTE LPAREN pattern RPAREN')
def quote_pattern(p):
    return [Symbol('quote'), p[2]]


@pg.production('receive_expr : RECEIVE COLON NEWLINE INDENT case_branches DEDENT')
def case(p):
    return [Symbol('match'), [Symbol('recv'), [Symbol('self')]]] + p[4]


@pg.production('record_expr : RECORD NAME')
def record_expr(p):
    return [Symbol('record'), token_to_symbol(p[1]), []]


@pg.production('record_expr : RECORD NAME OPLT NAME')
def record_expr(p):
    return [Symbol('record'), token_to_symbol(p[1]), token_to_symbol(p[3]), []]


@pg.production('record_expr : RECORD NAME LPAREN record_fields RPAREN')
def record_expr(p):
    return [Symbol('record'), token_to_symbol(p[1]), p[3]]


@pg.production('record_expr : RECORD NAME LPAREN record_fields RPAREN OPLT NAME')
def record_expr(p):
    return [Symbol('record'), token_to_symbol(p[1]), token_to_symbol(p[6]), p[3]]


@pg.production('record_expr : RECORD NAME COLON NEWLINE INDENT record_body DEDENT')
def record_expr(p):
    return [Symbol('record'), token_to_symbol(p[1]), []] + p[5]


@pg.production('record_expr : RECORD NAME OPLT NAME COLON NEWLINE INDENT record_body DEDENT')
def record_expr(p):
    return [Symbol('record'), token_to_symbol(p[1]), token_to_symbol(p[3]), []] + p[7]


@pg.production('record_expr : RECORD NAME LPAREN record_fields RPAREN COLON NEWLINE INDENT record_body DEDENT')
def record_expr(p):
    return [Symbol('record'), token_to_symbol(p[1]), p[3]] + p[8]


@pg.production('record_expr : RECORD NAME LPAREN record_fields RPAREN OPLT NAME COLON NEWLINE INDENT record_body DEDENT')
def record_expr(p):
    return [Symbol('record'), token_to_symbol(p[1]), token_to_symbol(p[6]), p[3]] + p[10]


@pg.production('record_body : def_expr')
def record_body(p):
    return [p[0]]


@pg.production('record_body : record_body def_expr')
def record_body(p):
    return p[0] + [p[1]]


@pg.production('record_fields : record_field')
def record_expr(p):
    return [p[0]]


@pg.production('record_fields : record_field COMMA record_fields')
def record_expr(p):
    return [p[0]] + p[2]


@pg.production('record_field : id_expr')
def record_expr(p):
    return p[0]


@pg.production('data_expr : DATA NAME COLON NEWLINE INDENT data_record_expr_list DEDENT')
def data_expr(p):
    return [Symbol('data'), token_to_symbol(p[1])] + p[5]


@pg.production('data_record_expr_list : data_record_expr')
def record_expr(p):
    return [p[0]]


@pg.production('data_record_expr_list : data_record_expr data_record_expr_list')
def record_expr(p):
    return [p[0]] + p[1]


@pg.production('data_record_expr : NAME LPAREN record_fields RPAREN NEWLINE')
def record_expr(p):
    return [token_to_symbol(p[0])] + p[2]


@pg.production('binop_expr : NOT binop_expr')
def binop_expr(p):
    return [token_to_symbol(p[0]), p[1]]


@pg.production('binop_expr : binop_expr OPPLUS binop_expr')
@pg.production('binop_expr : binop_expr OPMINUS binop_expr')
@pg.production('binop_expr : binop_expr OPTIMES binop_expr')
@pg.production('binop_expr : binop_expr PERCENT binop_expr')
@pg.production('binop_expr : binop_expr OPDIV binop_expr')
@pg.production('binop_expr : binop_expr OPLEQ binop_expr')
@pg.production('binop_expr : binop_expr OPGEQ binop_expr')
@pg.production('binop_expr : binop_expr OPEQ binop_expr')
@pg.production('binop_expr : binop_expr OPNEQ binop_expr')
@pg.production('binop_expr : binop_expr OPLT binop_expr')
@pg.production('binop_expr : binop_expr OPGT binop_expr')
@pg.production('binop_expr : binop_expr OPAND binop_expr')
@pg.production('binop_expr : binop_expr OPOR binop_expr')
@pg.production('binop_expr : binop_expr OPIS binop_expr')
def binop_expr(p):
    return [token_to_symbol(p[1]), p[0], p[2]]


@pg.production('binop_expr : binop_expr PIPELINE binop_expr')
def binop_expr(p):
    return [Symbol('|>'), p[0], p[2]]


@pg.production('binop_expr : binop_expr PIPELINE_BIND binop_expr')
def binop_expr(p):
    left, _, right = p
    input_sym = get_temp_name()
    return [Symbol('|>'), p[0], [Symbol('bind'),
                                 [Symbol('fn'), [input_sym], p[2] + [input_sym]]]]


@pg.production('binop_expr : binop_expr PIPELINE_FIRST binop_expr')
def binop_expr(p):
    return [Symbol('|>1'), p[0], p[2]]


@pg.production('binop_expr : binop_expr PIPELINE_FIRST_BIND binop_expr')
def binop_expr(p):
    left, _, right = p
    input_sym = get_temp_name()
    return [Symbol('|>'), p[0], [Symbol('bind'),
                                 [Symbol('fn'), [input_sym],
                                  [p[2][0], input_sym] + p[2][(1 if len(p[2]) > 1 else len(p[2])):]]]]


@pg.production('binop_expr : binop_expr PIPELINE_SEND binop_expr')
def binop_expr(p):
    return [Symbol('send'), p[0], p[2]]


@pg.production('binop_expr : binop_expr PIPELINE_MULTI_SEND binop_expr')
def binop_expr(p):
    return [Symbol('tuple'),
            [Symbol('filter'),
             [Symbol('fn'), [Symbol('$1')],
              [Symbol('send'), Symbol('$1'), p[2]]],
             p[0]]]


@pg.production('binop_expr : expr')
def binop_expr(p):
    return p[0]


name_seq = 0


def get_temp_name():
    global name_seq
    name_seq += 1
    name_symbol = Symbol('_gs%s' % name_seq)
    return name_symbol


REPL_CONTINUE = object()


def mod_lex(lexer, repl_mode=False):
    paren_openers = {'LPAREN', 'LBRACE', 'LBRACK'}
    paren_closers = {'RPAREN', 'RBRACE', 'RBRACK'}

    token_queue = Queue()
    indent_level = [0]
    ignore_newline = False
    paren_level = 0
    tab_len = 4

    def handle_newline(token):
        text = token.getstr()
        indent_str = text.rsplit('\n', 1)[1]
        indent = indent_str.count(' ') + indent_str.count('\t') * tab_len
        if indent > indent_level[-1]:
            indent_level.append(indent)
            indent_token = Token('INDENT', indent_str)
            indent_token.source_pos = token.getsourcepos()
            token_queue.put(indent_token)
        else:
            while indent < indent_level[-1]:
                indent_level.pop()
                dedent_token = Token('DEDENT', indent_str)
                token_queue.put(dedent_token)
        return token

    for token in lexer:
        while not token_queue.empty():
            queued_token = token_queue.get()
            if queued_token.gettokentype() in paren_openers:
                paren_level += 1
            elif queued_token.gettokentype() in paren_closers:
                paren_level -= 1
            ignore_newline = (paren_level > 0)
            yield queued_token

        if token.name == 'NAME':
            for rule in klg.rules:
                if rule.matches(token.value, 0):
                    token.name = rule.name
                    break
        elif token.gettokentype() == 'NEWLINE':
            if not ignore_newline:
                yield handle_newline(token)
            continue

        if token.gettokentype() in paren_openers:
            paren_level += 1
        elif token.gettokentype() in paren_closers:
            paren_level -= 1
        ignore_newline = (paren_level > 0)

        if token.gettokentype() == 'NAME' and token.getstr().startswith('&'):
            amp = Token('AMP', '&')
            amp.source_pos = token.getsourcepos()
            comma = Token('COMMA', ',')
            amp.source_pos = token.getsourcepos()
            name = Token('NAME', token.getstr()[1:])
            name.source_pos = token.getsourcepos()
            yield amp
            yield comma
            yield name
        else:
            yield token

    if repl_mode and len(indent_level) > 1:
        yield REPL_CONTINUE
    elif repl_mode and paren_level > 0:
        yield REPL_CONTINUE
    else:
        while len(indent_level) > 1:
            indent_level.pop()
            yield Token('DEDENT', '')

        while not token_queue.empty():
            yield token_queue.get()


def lex(input, repl_mode=False):
    return mod_lex(lg.build().lex(input), repl_mode)


def parse(lexer):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return pg.build().parse(lexer)
    except ParsingError as e:
        source_pos = e.getsourcepos()
        if source_pos is None:
            print('')
        else:
            print('ParsingError: lineno='
                  + str(source_pos.lineno)
                  + ' colno='
                  + str(source_pos.colno))
