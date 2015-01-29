from .utils import emit_sexp, issequence_except_str


class UnquoteSplicingError(Exception):
    def __init__(self):
        self.msg = 'unquote_splicing appeared in invalid context'

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
        self.msg = 'duplicated_def error: ' + \
                   'file "' + filename + '", ' + 'line ' + str(lineno) + ': ' + emit_sexp(exp)

    def __str__(self):
        return self.msg

    def __repr__(self):
        return self.msg


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