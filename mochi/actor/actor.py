import sys
from collections import Mapping, Set
from eventlet.queue import Queue
import eventlet
from eventlet.green import zmq
from urllib.parse import urlparse
from kazoo.client import KazooClient, KazooRetry
from msgpack import packb, unpackb, ExtType
from pyrsistent import (PVector, _PList, _PBag,
                        pvector, pmap, pset, plist, pbag)

_actor_map = {}

_actor_pool = eventlet.GreenPool()

_native_builtin_types = (int, float, str, bool)

TYPE_PSET = 1
TYPE_PLIST = 2
TYPE_PBAG = 3


def decode(obj):
    if isinstance(obj, ExtType):
        if obj.code == TYPE_PSET:
            unpacked_data = unpackb(obj.data,
                                    use_list=False,
                                    encoding='utf-8')
            return pset(decode(item) for item in unpacked_data)
        if obj.code == TYPE_PLIST:
            unpacked_data = unpackb(obj.data,
                                    use_list=False,
                                    encoding='utf-8')
            return plist(decode(item) for item in unpacked_data)
        if obj.code == TYPE_PBAG:
            unpacked_data = unpackb(obj.data,
                                    use_list=False,
                                    encoding='utf-8')
            return pbag(decode(item) for item in unpacked_data)
        module_name, class_name, *data = unpackb(obj.data,
                                                 use_list=False,
                                                 encoding='utf-8')
        cls = getattr(sys.modules[module_name],
                      class_name)
        return cls(*(decode(item) for item in data))
    if isinstance(obj, tuple):
        return pvector(decode(item) for item in obj)
    if isinstance(obj, dict):
        new_dict = dict()
        for key in obj.keys():
            new_dict[decode(key)] = decode(obj[key])
        return pmap(new_dict)
    return obj


def encode(obj):
    if type(obj) in (list, tuple) or isinstance(obj, PVector):
        return [encode(item) for item in obj]
    if isinstance(obj, Mapping):
        encoded_obj = {}
        for key in obj.keys():
            encoded_obj[encode(key)] = encode(obj[key])
        return encoded_obj
    if isinstance(obj, _native_builtin_types):
        return obj
    if isinstance(obj, Set):
        return ExtType(TYPE_PSET, packb([encode(item) for item in obj], use_bin_type=True))
    if isinstance(obj, _PList):
        return ExtType(TYPE_PLIST, packb([encode(item) for item in obj], use_bin_type=True))
    if isinstance(obj, _PBag):
        return ExtType(TYPE_PBAG, packb([encode(item) for item in obj], use_bin_type=True))
    # assume record
    cls = obj.__class__
    return ExtType(0, packb([cls.__module__, cls.__name__] + [encode(item) for item in obj],
                            use_bin_type=True))


class ActorAddressBook(object):
    def __init__(self, zk_hosts, timeout=60.0):
        self.retry = KazooRetry(max_tries=10)
        self.zk = KazooClient(hosts=zk_hosts, timeout=timeout)
        self.zk.start()

    def lookup(self, path):
        return self.retry(self._lookup, path)

    def _lookup(self, path):
        actor_url, stat = self.zk.get(path)
        return RemoteActor(actor_url.decode('utf-8'))

    def register(self, path, actor_url):
        return self.retry(self._register, path, actor_url)

    def _register(self, path, actor_url):
        self.zk.ensure_path(path)
        self.zk.set(path, actor_url.encode('utf-8'))

    def delete(self, path):
        self.zk.delete(path, recursive=True)

    def __del__(self):
        self.zk.stop()


class ActorHub(object):
    def __init__(self, url='tcp://*:9999'):
        self._path_actor_mapping = {}
        self._url = url
        self._context = zmq.Context()
        self._recv_sock = self._context.socket(zmq.PAIR)
        self._recv_sock.bind(url)

    def register(self, path, actor):
        if isinstance(actor, ActorBase):
            self._path_actor_mapping[path] = actor
        else:
            raise TypeError('can only register an actor')

    def unregister(self, path):
        del self._path_actor_mapping[path]

    def _run(self):
        while True:
            pyobj = decode(unpackb(self._recv_sock.recv(),
                                   encoding='utf-8',
                                   use_list=False))
            path, msg = pyobj
            if path in self._path_actor_mapping:
                self._path_actor_mapping[path].send(msg)
            else:
                # ignore
                pass

    def run(self):
        _actor_pool.spawn(self._run)

    def __del__(self):
        self._recv_sock.close()


class ActorBase(object):
    pass


class RemoteActor(ActorBase):

    def __init__(self, url):
        parsed_url = urlparse(url)
        self._base_url = parsed_url.scheme + "://" + parsed_url.netloc
        self._path = parsed_url.path[1:]
        self._context = zmq.Context()
        self._send_sock = self._context.socket(zmq.PAIR)
        self._send_sock.connect(self._base_url)

    def send(self, msg):
        self._send_sock.send(packb(encode((self._path, msg)),
                                   encoding='utf-8',
                                   use_bin_type=True))

    def __del__(self):
        self._send_sock.close()


class Actor(ActorBase):
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
    if isinstance(actor, ActorBase):
        actor.send(message)
    else:
        default_inbox.put(message)


def recv(actor=None):
    if isinstance(actor, ActorBase):
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
