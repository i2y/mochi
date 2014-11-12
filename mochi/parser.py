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

lg.add('SQUOTE_STR', r"(?x)'(?:|[^'\\]|\\.|\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8})*'")
lg.add('DQUOTE_STR', r'(?x)"(?:|[^"\\]|\\.|\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8})*"')

lg.add('UNTERMINATED_STRING', r"[\"\'].*")
lg.add('NUMBER', r'-?[0-9]+(?:\.[0-9]+)?')
lg.add('NAME', r'\&?[_a-zA-Z$][-_a-zA-Z0-9]*')
lg.add('OPPAREN', r'" + operator_regex_str + "(?=\()')
lg.add('PIPELINE', r'\|>')
lg.add('BAR', r'\|')
lg.add('LBRACK', r'\[')
lg.add('RBRACK', r'\]')
lg.add('LBRACE', r'\{')
lg.add('RBRACE', r'\}')
lg.add('LPAREN', r'\(')
lg.add('RPAREN', r'\)')
lg.add('DOT', r'\.')
lg.add('PERCENT', r'%')
lg.add('COMMA', r',')
lg.add('THINARROW', r'->')
lg.add('COLONCOLON', r'::')
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

lg.add('NEWLINE', r'\r?\n[\t ]*')
lg.ignore(r'[ \t\f\v]+')
lg.ignore(r'#.*(?:\n|\r|\r\n|\n\r|$)')  # comment

klg = LexerGenerator()
klg.add('IMPORT', r'^import$')
klg.add('MODULE', r'^module$')
klg.add('REQUIRE', r'^require$')
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
klg.add('THENCOLON', r'^then:$')
klg.add('ELSE', r'^else$')
klg.add('ELSEIF', r'^elif$')
klg.add('MATCH', r'^match$')
klg.add('OF', r'^of$')
klg.add('RECORD', r'^record$')
klg.add('DATA', r'^data$')
klg.add('YIELD', r'^yield$')
klg.add('RETURN', r'^return$')
klg.add('WITH:', r'^with$')
klg.add('MUTABLE', r'^mutable$')
klg.add('DATATYPE', r'^datatype$')
klg.add('FOR', r'^for$')
klg.add('IN', r'^in$')
klg.add('FROM', r'^from$')
klg.add('END', r'^end$')
klg.add('LAZY', r'^lazy$')
klg.add('OPAND', r'^and$')
klg.add('OPOR', r'^or$')
klg.add('OPIS', r'^is$')
klg.add('NOT', r'^not$')

pg = ParserGenerator(['NUMBER', 'OPPLUS', 'OPMINUS', 'OPTIMES', 'OPDIV', 'OPLEQ', 'OPGEQ', 'OPEQ', 'OPNEQ',
                      'OPLT', 'OPGT', 'OPAND', 'OPOR', 'OPIS', 'NOT', 'NEWLINE', 'PERCENT',
                      'LPAREN', 'RPAREN', 'TRUE', 'FALSE', 'DQUOTE_STR', 'SQUOTE_STR', 'AT',
                      'NAME', 'EQUALS', 'IF', 'ELSEIF', 'ELSE', 'COLON', 'SEMI', 'DATA', 'IMPORT', 'REQUIRE',
                      'LBRACK', 'RBRACK', 'COMMA', 'DEF', 'DOC', 'CALET', 'PIPELINE', 'RETURN',
                      'LBRACE', 'RBRACE', 'MATCH', 'DEFM', 'RECORD', 'AMP', 'FN', 'THINARROW',
                      'YIELD', 'FROM', 'FOR', 'IN', 'DOT', 'INDENT', 'DEDENT', 'TRY', 'FINALLY', 'EXCEPT',
                      'MODULE', 'AS', 'RAISE'],
                     precedence=[('left', ['EQUALS']),
                                 ('left', ['NOT']),
                                 ('left', ['OPIS']),
                                 ('left', ['OPEQ', 'OPLEQ', 'OPGEQ', 'OPNEQ', 'OPLT', 'OPGT', 'OPAND', 'OPOR']),
                                 ('left', ['OPPLUS', 'OPMINUS']),
                                 ('left', ['LBRACK', 'RBRACK']),
                                 ('left', ['OPTIMES', 'OPDIV', 'PERCENT'])],
                     cache_id='klon')


@pg.production('program : block')
def program(p):
    return p[0]


@pg.production('block : stmts')
def block(p):
    return p[0]


@pg.production('stmts : stmts stmt')
def stmts_b(p):
    # if issequence_except_str(p[1]):
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
# @pg.production('stmt : if_expr')
@pg.production('stmt : try_stmt')
@pg.production('stmt : raise_stmt')
@pg.production('stmt : return_stmt')
def stmt(p):
    return p[0]


@pg.production('import_expr : IMPORT NAME')
def import_expr(p):
    return [Symbol('import'), token_to_symbol(p[1])]


@pg.production('require_expr : REQUIRE string')
def require_expr(p):
    return [Symbol('require'), p[1]]


@pg.production('module_expr : MODULE NAME COLON NEWLINE INDENT export_cls block DEDENT')
def module_expr(p):
    return [Symbol('module'), token_to_symbol(p[1]), p[5]] + p[6]


@pg.production('export_cls : names')
def export_cls(p):
    return p[0]


@pg.production('names : names COMMA name')
def names(p):
    return p[0] + [p[2]]


@pg.production('names : name')
def names_single(p):
    return [p[0]]


@pg.production('name : NAME')
def name(p):
    return token_to_symbol(p[0])


@pg.production('tuple_elt : binop_expr COMMA')
def tuple_elt(p):
    return p[0]


@pg.production('from_expr : FROM NAME IMPORT NAME')
def from_expr(p):
    return [Symbol('from-import'), token_to_symbol(p[1]), token_to_symbol(p[3])]


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


@pg.production('return_stmt : RETURN binop_expr')
def raise_stmt(p):
    return [Symbol('return'), p[1]]


def token_to_symbol(token):
    return Symbol(token.getstr(), token.getsourcepos().lineno, token.getsourcepos().colno)


def token_to_keyword(token):
    return Keyword(token.getstr(), token.getsourcepos().lineno, token.getsourcepos().colno)


@pg.production('let_expr : pattern EQUALS binop_expr')
def let_expr(p):
    return [Symbol('val', 0, 0), p[0], p[2]]


@pg.production('binding : NAME')
def binding(p):
    return token_to_symbol(p[0])


@pg.production('expr : fn_expr')
@pg.production('expr : paren_expr')
@pg.production('expr : if_expr')
@pg.production('expr : prim_expr')
@pg.production('expr : app_expr')
@pg.production('expr : left_app_expr')
@pg.production('expr : right_app_expr')
@pg.production('expr : dict_expr')
@pg.production('expr : tuple_expr')
@pg.production('expr : match_expr')
@pg.production('expr : yield_expr')
@pg.production('expr : yield_from_expr')
@pg.production('expr : for_expr')
@pg.production('expr : block_expr')
@pg.production('expr : defm_expr')
@pg.production('expr : dot_expr')
@pg.production('expr : id_expr')
# @pg.production('expr : app_nc_expr')
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
def expr_dquote_str(p):
    return p[0].getstr()[1:-1]


@pg.production('string : SQUOTE_STR')
def expr_squote_str(p):
    return p[0].getstr()[1:-1]


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


@pg.production('yield_expr : YIELD binop_expr')
def yield_expr(p):
    return [Symbol('yield'), p[1]]


@pg.production('yield_from_expr : YIELD FROM binop_expr')
def yield_from_expr(p):
    return [Symbol('yield-from'), p[1]]


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


@pg.production('dot_expr : expr DOT NAME')
def dot_expr(p):
    return [Symbol('getattr'), p[0], p[2].getstr()]


@pg.production('for_expr : LBRACK binop_expr FOR pattern IN binop_expr RBRACK')  # TODO
def for_expr(p):
    pattern = p[3]
    items = p[5]
    body = p[1]
    return [Symbol('tuple-of')] + [body] + [[pattern, items]]


@pg.production('tuple_expr : LBRACK tuple_elts binop_expr RBRACK')
def tuple_expr(p):
    return [Symbol('make-tuple')] + p[1] + [p[2]]


@pg.production('tuple_expr : LBRACK binop_expr RBRACK')
def tuple_expr_one(p):
    return [Symbol('make-tuple'), p[1]]


@pg.production('tuple_expr : LBRACK tuple_elts binop_expr RBRACK')
def tuple_expr(p):
    return [Symbol('make-tuple')] + p[1] + [p[2]]


@pg.production('tuple_expr : LBRACK binop_expr RBRACK')
def tuple_expr_one(p):
    return [Symbol('make-tuple'), p[1]]


@pg.production('tuple_expr : LBRACK RBRACK')
def tuple_expr_empty(p):
    return [Symbol('make-tuple')]


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
    return [Symbol('with-decorator')] + p[0] + [p[1]]


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


@pg.production('defm_expr : DEFM id_expr doc_string COLON NEWLINE INDENT case_branches DEDENT')
def fun_expr(p):
    return [Symbol('defm'), p[1]] + p[6]


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


@pg.production('app_args_elt : binop_expr')
def app_args_elt(p):
    return [p[0]]


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


@pg.production('right_app_expr : expr PIPELINE right_app_fun_expr app_args')
def right_app_expr(p):
    expr, _, right_app_fun_expr, app_args = p
    return [right_app_fun_expr] + app_args + [expr]


@pg.production('right_app_fun_expr : id_expr')
def right_app_fun_expr(p):
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


# TODO
# @pg.production('pattern : id_expr')
# @pg.production('pattern : tuple_expr')
# @pg.production('pattern : dict_expr')
@pg.production('pattern : binop_expr')
def pattern(p):
    return p[0]


@pg.production('record_expr : RECORD id_expr')
def record_expr(p):
    return [Symbol('record'), p[1], []]


@pg.production('record_expr : RECORD id_expr LPAREN record_fields RPAREN')
def record_expr(p):
    return [Symbol('record'), p[1], p[3]]


@pg.production('record_fields : record_field')
def record_expr(p):
    return [p[0]]


@pg.production('record_fields : record_field COMMA record_fields')
def record_expr(p):
    return [p[0]] + p[2]


@pg.production('record_field : id_expr')
def record_expr(p):
    return p[0]


@pg.production('data_expr : DATA id_expr COLON NEWLINE INDENT data_record_expr_list DEDENT')
def data_expr(p):
    return [Symbol('data'), p[1]] + p[5]


@pg.production('data_record_expr_list : data_record_expr')
def record_expr(p):
    return [p[0]]


@pg.production('data_record_expr_list : data_record_expr data_record_expr_list')
def record_expr(p):
    return [p[0]] + p[1]


@pg.production('data_record_expr : id_expr LPAREN record_fields RPAREN NEWLINE')
def record_expr(p):
    return [p[0]] + p[2]


@pg.production('binop_expr : NOT expr')
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


@pg.production('binop_expr : expr')
def binop_expr(p):
    return p[0]


REPL_CONTINUE = object()


def mod_lex(lexer, repl_mode=False):
    paren_openers = 'LPAREN', 'LBRACE', 'LBRACK'
    paren_closers = 'RPAREN', 'RBRACE', 'RBRACK'

    token_queue = []
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
            token_queue.append(indent_token)
        else:
            while indent < indent_level[-1]:
                indent_level.pop()
                dedent_token = Token('DEDENT', indent_str)
                token_queue.append(dedent_token)
        return token

    for token in lexer:
        if token.name == 'NAME':
            for rule in klg.rules:
                if rule.matches(token.value, 0):
                    token.name = rule.name
                    break

        while len(token_queue) > 0:
            tmp = token_queue.pop()
            yield tmp

        if token.gettokentype() == 'NEWLINE':
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
    else:
        while len(indent_level) > 1:
            indent_level.pop()
            yield Token('DEDENT', '')

        for token in token_queue:
            yield token


def lex(input, repl_mode=False):
    return mod_lex(lg.build().lex(input), repl_mode)


def parse(lexer):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return pg.build().parse(lexer)
    except ParsingError as e:
        print("ParsingError: lineno=" + str(e.getsourcepos().lineno) + " colno=" + str(e.getsourcepos().colno))
