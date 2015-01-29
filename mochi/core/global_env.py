
from collections import Iterable, Mapping, MutableMapping, Sequence, MutableSequence
import functools
import itertools
from numbers import Number
import operator
import re

from pyrsistent import (v, pvector, m, pmap, s, pset, b, pbag, dq, pdeque, l,
                        plist, pclass, freeze, thaw)

from mochi import IS_PYPY
from mochi.actor import actor
from mochi.parser.parser import Symbol, Keyword, get_temp_name


def make_default_env():

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
    env['seq_count'] = MutableSequence.count
    env['extend'] = MutableSequence.extend
    env['insert'] = MutableSequence.insert
    env['pop'] = MutableSequence.pop
    env['remove'] = MutableSequence.remove
    env['reverse'] = MutableSequence.reverse
    env['mapping_get'] = MutableMapping.get
    env['items'] = MutableMapping.items
    env['values'] = MutableMapping.values
    env['keys'] = MutableMapping.keys
    env['mapping_pop'] = MutableMapping.pop
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
    env['gensym'] = get_temp_name
    env['uniq'] = get_temp_name
    env['Record'] = pclass((), 'Record')  # namedtuple('Record', ())
    env['spawn'] = actor.spawn
    env['send'] = actor.send
    env['recv'] = actor.recv
    env['link'] = actor.link
    env['unlink'] = actor.unlink
    env['kill'] = actor.kill
    env['cancel'] = actor.cancel
    env['self'] = actor.self
    env['sleep'] = actor.sleep
    env['wait_all'] = actor.wait_all
    env['wait'] = actor.wait
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