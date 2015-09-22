from abc import ABCMeta, abstractmethod
import uuid

import eventlet

from .message import (Down, Monitor, Unmonitor, Cancel, Kill, Fork,
                      ForkWithMonitor, ForkResponse)
from .mailbox import Mailbox, AckableMailbox, LocalMailbox, Receiver


_actor_map = {}

_actor_pool = eventlet.GreenPool(size=1000000)


class ActorBase(Receiver, metaclass=ABCMeta):
    @abstractmethod
    def encode(self):
        pass

    @staticmethod
    @abstractmethod
    def decode(params):
        pass


class Actor(ActorBase):
    __slots__ = ['_ack', '_inbox', '_outbox',
                 '_callback', '_greenlet', '_observers']

    def __init__(self, callback, mailbox=None):
        if mailbox is None:
            mailbox = LocalMailbox()
        assert isinstance(mailbox, Mailbox)
        self._ack = isinstance(mailbox, AckableMailbox)
        self._inbox = mailbox
        self._outbox = mailbox
        self._callback = callback
        self._greenlet = None
        self._observers = {}

    def run(self, *args, **kwargs):
        greenlet_id = id(eventlet.getcurrent())
        _actor_map[greenlet_id] = self
        try:
            self._callback(*args, **kwargs)
        finally:
            if greenlet_id in _actor_map.keys():
                del _actor_map[greenlet_id]

    def spawn(self, *args, **kwargs):
        self._greenlet = _actor_pool.spawn(self.run, *args, **kwargs)

    def _link(self, func, *args, **kwargs):
        if self._greenlet is None:
            return
        return self._greenlet.link(func, *args, **kwargs)

    def _unlink(self, func, *args, **kwargs):
        if self._greenlet is None:
            return
        return self._greenlet.unlink(func, *args, **kwargs)

    def _cancel(self, *throw_args):
        if self._greenlet is None:
            return
        return self._greenlet.cancel(*throw_args)

    def _kill(self, *throw_args):
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
        if self._inbox is None:
            raise RuntimeError('mailbox is None')

        while True:
            message = self._inbox.get()
            if isinstance(message, Monitor):
                try:
                    self._observers[message.sender] = spawn(observe,
                                                            self,
                                                            message.sender)
                finally:
                    self.ack_last_msg()
                continue
            elif isinstance(message, Unmonitor):
                try:
                    self._observers[message.sender]._kill()
                    del self._observers[message.sender]
                finally:
                    self.ack_last_msg()
                continue
            elif isinstance(message, Cancel):
                try:
                    self._cancel()
                finally:
                    self.ack_last_msg()
                continue
            elif isinstance(message, Kill):
                try:
                    self._kill()
                finally:
                    self.ack_last_msg()
                continue
            elif isinstance(message, Fork):
                try:
                    new_actor = spawn_with_mailbox(message.func,
                                                   self._inbox,
                                                   *message.args,
                                                   **message.kwargs)
                    send(ForkResponse(new_actor), message.sender)
                    self._kill()
                finally:
                    self.ack_last_msg()
                continue
            elif isinstance(message, ForkWithMonitor):
                try:
                    new_actor = spawn_with_mailbox(message.func,
                                                   self._inbox,
                                                   *message.args,
                                                   **message.kwargs)
                    self._observers[message.sender] = spawn(observe,
                                                            new_actor,
                                                            message.sender)
                    send(ForkResponse(new_actor), message.sender)
                    self._kill()
                finally:
                    self.ack_last_msg()
                continue
            return message

    def ack_last_msg(self):
        if self._ack:
            self._inbox.ack()

    def encode(self):
        return self._inbox.encode()

    @staticmethod
    def decode(params):
        raise NotImplementedError

    def __del__(self):
        del self._observers

    def __str__(self):
        return str(id(self._greenlet)) + '@' + str(self._inbox)

    def __eq__(self, other):
        return (self.__class__ is other.__class__
                and self._inbox == other._inbox)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(str(self))


def observe(target, observer):
    try:
        exit_value = target.wait()
        send(Down(target, exit_value), observer)
    except Exception as e:
        send(Down(target, {'exception name:': e.__class__.__name__,
                           'args': e.args}),
             observer)


def make_ref():
    return uuid.uuid4()

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


def link(receiver: Receiver):
    send(Monitor(self()), receiver)
    return

monitor = link


def unlink(receiver: Receiver):
    send(Unmonitor(self()), receiver)
    return

unmonitor = unlink


def cancel(receiver: Receiver):
    send(Cancel(self()), receiver)
    return


def kill(receiver: Receiver):
    send(Kill(self()), receiver)
    return


def fork(receiver: Receiver, func, *args, **kwargs):
    current_actor = self()
    send(Fork(current_actor, func, args, kwargs), receiver)
    while True:
        message = recv(current_actor)
        if isinstance(message, ForkResponse):
            return message.new_actor
        else:
            send(message, current_actor)
    return


def fork_with_monitor(receiver: Receiver, func, *args, **kwargs):
    current_actor = self()
    send(ForkWithMonitor(current_actor, func, args, kwargs), receiver)
    while True:
        message = recv(current_actor)
        if isinstance(message, ForkResponse):
            return message.new_actor
        else:
            send(message, current_actor)
    return


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
