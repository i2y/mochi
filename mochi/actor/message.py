from pyrsistent import immutable


class Monitor(immutable('sender', name='_Monitor')):
    pass


class Unmonitor(immutable('sender', name='_Unmonitor')):
    pass


class Cancel(immutable('sender', name='_Cancel')):
    pass


class Kill(immutable('sender', name='_Kill')):
    pass


class Fork(immutable('sender, func, args, kwargs', name='_Fork')):
    pass


class ForkWithMonitor(immutable('sender, func, args, kwargs', name='_ForkWithMonitor')):
    pass


class ForkResponse(immutable('new_actor ', name='_ForkResponse')):
    pass


class Down(immutable('sender, reason', name='_OkMessage')):
    pass
