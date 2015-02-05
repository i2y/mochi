import ast
from collections import Iterable, Sequence, MutableSequence
from functools import reduce
from pathlib import Path
from os import chdir
from os.path import normpath, abspath

from .utils import issequence_except_str
from .constants import *
from .exceptions import MochiSyntaxError, DuplicatedDefError
from .global_env import global_env
from mochi import GE_PYTHON_34, IS_PYPY
from mochi.parser.parser import Symbol, Keyword, parse, lex, get_temp_name


if GE_PYTHON_34:
    expr_and_stmt_ast = (ast.Expr, ast.If, ast.For, ast.FunctionDef, ast.Assign, ast.Delete, ast.Try, ast.Raise,
                         ast.With, ast.While, ast.Break, ast.Return, ast.Continue, ast.ClassDef,
                         ast.Import, ast.ImportFrom, ast.Pass)
else:
    expr_and_stmt_ast = (ast.Expr, ast.If, ast.For, ast.FunctionDef, ast.Assign, ast.Delete, ast.TryFinally,
                         ast.TryExcept, ast.Raise, ast.With, ast.While, ast.Break, ast.Return, ast.Continue,
                         ast.ClassDef, ast.Import, ast.ImportFrom, ast.Pass)

syntax_table = {}


#-- double builtins :(
def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def tuple_it(obj):
    return tuple(map(tuple_it, obj)) if isinstance(obj, MutableSequence) else obj

#---

def syntax(*names):
    def _syntax(func):
        for name in names:
            syntax_table[name] = func
            if name not in global_env.keys():
                global_env[name] = 'syntax_' + name
        return func

    return _syntax

required_filenames_stack = [set()]


def lexical_scope_for_require(func):
    def _lexical_scope(*args, **kwargs):
        try:
            required_filenames_stack.append(set())
            return func(*args, **kwargs)
        finally:
            required_filenames_stack.pop()
    return _lexical_scope


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


def add_binding_name(binding_name, file_name):
    return binding_name_set_stack[-1].add_binding_name(binding_name, file_name)


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
        required_filenames = required_filenames_stack[-1]
        if filename in required_filenames:
            return self.translate_ref(NONE_SYM)[1],
        else:
            required_filenames.add(filename)
            return self.translate_loaded_file(filename)

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

    @syntax('yield_from')
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
            'is', 'is_not', 'in', 'not_in')
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

    @syntax('with_decorator')
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
                        arg_name = ast.Name(id=vararg.arg if GE_PYTHON_34 else vararg,
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

        exp = exps[-1]
        if not issequence_except_str(exp):
            exp = (Symbol('return'), exp)
        elif not isinstance(exp[0], Symbol):
            exp = (Symbol('return'), exp)
        elif exp[0].name in {'val', '_val', 'return', 'yield', 'yield_from', 'pass', 'raise'}:
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
        elif exp[0].name in {'do', 'with', 'finally', 'except'}:
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
        if GE_PYTHON_34:
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

    @syntax('with')
    def translate_with(self, exp):
        if len(exp) < 3:
            raise MochiSyntaxError(exp, self.filename)

        if GE_PYTHON_34:
            return self.translate_with_34(exp)
        else:
            return self.translate_with_old(exp)

    def translate_with_34(self, exp):
        keyword_with, items, *body = exp
        pre = []
        items_py = []
        for item in items:
            item_pre, item_value = self.translate(item[0], False)
            pre.extend(item_pre)
            var = item[1]
            items_py.append(ast.withitem(context_expr=item_value,
                                         optional_vars=ast.Name(id=var.name,
                                                                ctx=ast.Store(),
                                                                lineno=var.lineno,
                                                                col_offset=0),
                                         lineno=var.lineno,
                                         col_offset=0))

        body_py = self._translate_sequence(body, True)
        pre.append(ast.With(items=items_py,
                            body=body_py,
                            lineno=keyword_with.lineno,
                            col_offset=0))
        return pre, self.translate(NONE_SYM, False)[1]

    def translate_with_old(self, exp):
        keyword_with, items, *body = exp
        pre = []
        first_with_py = None
        with_py = None
        for item in items:
            item_pre, item_value = self.translate(item[0], False)
            pre.extend(item_pre)
            var = item[1]
            if with_py is None:
                with_py = ast.With(context_expr=item_value,
                                   optional_vars=ast.Name(id=var.name,
                                                          ctx=ast.Store(),
                                                          lineno=var.lineno,
                                                          col_offset=0),
                                   lineno=var.lineno,
                                   col_offset=0)
                first_with_py = with_py
            else:
                inner_with_py = ast.With(context_expr=item_value,
                                         optional_vars=ast.Name(id=var.name,
                                                                ctx=ast.Store(),
                                                                lineno=var.lineno,
                                                                col_offset=0),
                                         lineno=var.lineno,
                                         col_offset=0)
                with_py.body = [inner_with_py]
                with_py = inner_with_py

        with_py.body = self._translate_sequence(body, True)
        pre.append(first_with_py)
        return pre, self.translate(NONE_SYM, False)[1]

    @syntax('raise')
    def translate_raise(self, exp):
        if len(exp) != 2:
            raise MochiSyntaxError(exp, self.filename)
        return (ast.Raise(exc=self.translate(exp[1], False)[1],
                          cause=None,
                          lineno=exp[0].lineno,
                          col_offset=0),), self.translate(EMPTY_SYM, False)[1]

    @syntax('def')
    @lexical_scope_for_require
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
                             annotation=None) if GE_PYTHON_34 else vararg_exp.name

        if varkwarg_exp is None:
            varkwarg = None
        else:
            varkwarg = ast.arg(arg=varkwarg_exp.name,
                               annotation=None) if GE_PYTHON_34 else varkwarg_exp.name

        # 末尾のS式をreturnに変換する。
        # 末尾がif式やdo式の場合はその中まで変換する。
        exp = tuple(exp[:3]) + self._tail_to_return_s(exp[3:])  # TODO 最後が変数参照だと戻り値がNoneになる。
        body = self._translate_sequence(exp[3:])

        if varkwarg is not None:
            node = ast.parse('{0} = pmap({0})'.format(varkwarg.arg if GE_PYTHON_34 else varkwarg))
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

        if parent_record is not RECORD_SYM:
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
        else:
            record_def = ast.ClassDef(name=record_name.name,
                                      bases=[ast.Call(func=ast.Name(id='pclass',
                                                                    lineno=lineno,
                                                                    col_offset=col_offset,
                                                                    ctx=ast.Load()),
                                                      args=[
                                                          ast.Tuple(elts=members,
                                                                    lineno=parent_record.lineno,
                                                                    col_offset=parent_record.col_offset,
                                                                    ctx=ast.Load()),
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
        names = [ast.alias(name=import_sym.name,
                           asname=None,
                           lineno=import_sym.lineno,
                           col_offset=import_sym.col_offset) for import_sym in exp[1:]]
        return (ast.Import(names=names,
                           lineno=exp[0].lineno,
                           col_offset=exp[0].col_offset),), self.translate_ref(NONE_SYM)[1]

    @syntax('from_import')
    def translate_from(self, exp):
        if len(exp) < 3:
            raise MochiSyntaxError(exp, self.filename)
        names = [ast.alias(name=import_sym.name,
                           asname=None,
                           lineno=import_sym.lineno,
                           col_offset=import_sym.col_offset) for import_sym_names in exp[2:]
                                                             for import_sym in import_sym_names]
        return (ast.ImportFrom(module=exp[1].name,
                               names=names,
                               level=0,
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

    @syntax('make_tuple')
    @syntax('make_pvector')
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

    @syntax('make_list')
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
                elif value[0].name == 'unquote_splicing':
                    value1_pre, value1_body = self.translate(value[1], False)
                    return (tuple(value1_pre),
                            [self.translate((Symbol('quote'),
                                             Symbol('_mochi_unquote_splicing')), False), value1_body])
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

# 'translator' and 'bindingNameSetStack' is a global variable.
translator = Translator()
global_scope = Scope()
for name in global_env.keys():
    global_scope.add_binding_name(name, "<builtin>")
binding_name_set_stack = [global_scope]
