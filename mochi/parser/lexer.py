import sys
from queue import Queue

from rply import LexerGenerator, Token

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


def lex(input, repl_mode=False, debug=True):  # TODO
    if debug:
        print(list(mod_lex(lg.build().lex(input), repl_mode)), file=sys.stderr)

    return mod_lex(lg.build().lex(input), repl_mode)


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
klg.add('VECTOR', r'^vector$')
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
