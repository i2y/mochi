from eventlet.queue import Queue
import eventlet

_actor_map = {}

_actor_pool = eventlet.GreenPool()


class Actor(object):
    def __init__(self, callback):
        self._inbox = Queue()
        self._callback = callback
        self._greenlet = None

    def run(self, *args, **kwargs):
        greenlet_id = id(eventlet.getcurrent())
        _actor_map[greenlet_id] = self
        try:
            self._callback(*args, **kwargs)
        finally:
            del _actor_map[greenlet_id]

    def spawn(self, *args, **kwargs):
        self._greenlet = _actor_pool.spawn(self.run, *args, **kwargs)

    def link(self, func, *args, **kwargs):
        if self._greenlet is None:
            return
        return self._greenlet.link(func, *args, **kwargs)

    def unlink(self, func, *args, **kwargs):
        if self._greenlet is None:
            return
        return self._greenlet.unlink(func, *args, **kwargs)

    def cancel(self, *throw_args):
        if self._greenlet is None:
            return
        return self._greenlet.cancel(*throw_args)

    def kill(self, *throw_args):
        if self._greenlet is None:
            return
        return self._greenlet.kill(*throw_args)

    def wait(self):
        if self._greenlet is None:
            return
        return self._greenlet.wait()

    def send(self, message):
        self._inbox.put(message)

    def receive(self):
        return self._inbox.get()


default_inbox = Queue()


def send(message, actor=None):
    if isinstance(actor, Actor):
        actor.send(message)
    else:
        default_inbox.put(message)


def recv(actor=None):
    if isinstance(actor, Actor):
        return actor.receive()
    else:
        return default_inbox.get()


def link(actor, func, *args, **kwargs):
    return actor.link(func, *args, **kwargs)


def unlink(actor, func, *args, **kwargs):
    return actor.unlink(func, *args, **kwargs)


def cancel(actor, *throw_args):
    return actor.cancel(*throw_args)


def kill(actor, *throw_args):
    return actor.kill(*throw_args)


def self():
    cur_green = eventlet.getcurrent()
    return _actor_map.get(id(cur_green))


def spawn(func, *args, **kwargs):
    actor = Actor(func)
    actor.spawn(*args, **kwargs)
    return actor


def sleep(seconds):
    eventlet.sleep(seconds)


def wait_all():
    _actor_pool.waitall()


def wait(actor):
    return actor.wait()
