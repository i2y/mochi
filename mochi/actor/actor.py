from abc import ABCMeta

import eventlet
from .mailbox import Mailbox, AckableMailbox, LocalMailbox


_actor_map = {}

_actor_pool = eventlet.GreenPool()


class ActorBase(metaclass=ABCMeta):
    pass


class Actor(ActorBase):
    def __init__(self, callback, mailbox=LocalMailbox()):
        assert isinstance(mailbox, Mailbox)
        self._ack = isinstance(mailbox, AckableMailbox)
        self._inbox = mailbox
        self._outbox = mailbox
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
        if self._outbox is not None:
            self._outbox.put(message)

    def receive(self):
        if self._inbox is not None:
            return self._inbox.get()

    def ack_last_msg(self):
        if self._ack:
            self._inbox.ack()

default_mailbox = LocalMailbox()


def send(message, target=None):
    if isinstance(target, ActorBase):
        target.send(message)
    elif isinstance(target, Mailbox):
        target.put(message)
    else:
        default_mailbox.put(message)


def recv(target=None):
    if isinstance(target, ActorBase):
        return target.receive()
    elif isinstance(target, Mailbox):
        return target.get()
    else:
        return default_mailbox.get()


def ack_last_msg(target=None):
    if isinstance(target, Actor):
        target.ack_last_msg()
    elif isinstance(target, AckableMailbox):
        target.ack()


def ack():
    self().ack_last_msg()


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


def spawn_with_mailbox(func, mailbox, *args, **kwargs):
    actor = Actor(func, mailbox)
    actor.spawn(*args, **kwargs)
    return actor


def sleep(seconds):
    eventlet.sleep(seconds)


def wait_all():
    _actor_pool.waitall()


def wait(actor):
    return actor.wait()
