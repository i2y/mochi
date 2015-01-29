from collections import Sequence
from .constants import LPARA, RPARA


def issequence(obj):
    return isinstance(obj, Sequence)


def issequence_except_str(obj):
    if isinstance(obj, str):
        return False
    return isinstance(obj, Sequence)


def is_tuple_or_list(obj):
    return type(obj) in {tuple, list}


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