from eventlet.queue import Queue
import eventlet

_actor_map = {}

_actor_pool = eventlet.GreenPool()


class Actor(object):
    def __init__(self, callback):
        self._inbox = Queue()
        self._callback = callback

    def run(self):
        cur_green = eventlet.getcurrent()
        _actor_map[id(cur_green)] = self # TODO 後片付け
        self._callback()

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


def self():
    cur_green = eventlet.getcurrent()
    return _actor_map.get(id(cur_green))


def spawn(func):
    actor = Actor(func)
    _actor_pool.spawn(actor.run)
    return actor


def sleep(seconds):
    eventlet.sleep(seconds)


def wait_all():
    _actor_pool.waitall()
