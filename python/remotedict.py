"""
Provides remote dictionary like storage object.

Usage:

  import remotedict as rd

  # run the following in server or client host
  server = rd.Server().run()

  # run the following in client host or in several clients
  d = rd.Storage()  # d is dict-like object
  d['a'] = 1
  d.keys()    # -> ('a', )
  d.clear()


"""
# Author: Pearu Peterson
# Created: November 2018

# TODO: implement persistence of Server cache

import uuid
import socket
import hashlib
try:
    import dill as pickle
except ImportError as msg:
    print('Failed to import dill:', msg,
          ' (use `conda!pip install dill` to install). Falling back to pickle.')
    import pickle
import traceback
import sys
import time
from collections import defaultdict

magic = b'storageserver'

default_encoding = 'utf-8'
default_host = 'localhost'
default_port = 36610
server_timeout = None


def digest(data):
    """Return data digest.
    """
    m = hashlib.sha256()
    m.update(data)
    return m.hexdigest()    


def make_message(action, key, data):
    """Compose a message.

    Parameters
    ----------
    action : str
      Specify message action
    key : str
      Specify message key
    data : bytes
      Specify message data

    Returns
    -------
    message : bytes
    """
    assert isinstance(data, bytes), repr(type(data))
    nbytes = len(data)
    header = ':{nbytes}:{action}:{key}:'.format_map(locals())
    assert len(header) < 1024, repr(len(header))
    message = magic + header.encode() + data
    return message


def read_header(message):
    """Extract message header data

    Parameters
    ----------
    message : bytes
      Specify message

    Returns
    -------
    nbytes : int
      Expected data size in bytes
    action : str
      Message action
    key : str
      Message key
    data : bytes
      Data head

    """
    parts = message.split(b':', 4)
    assert len(parts) == 5, repr((len(parts), parts))
    magic_, nbytes, action, key, data = parts
    sys.stdout.flush()
    assert magic_ == magic, repr((magic_, magic))
    nbytes = int(nbytes)
    return nbytes, action.decode(), key.decode(), data


def read_message(sock, bufsize=1024):
    """Read message from socket

    Parameters
    ----------
    sock : socket.Socket
      Specify open socket
    bufsize : int
      Specify read buffer size.

    Returns
    -------
    action : str
      Message action
    key : str
      Message key
    data : bytes
      Message data
    """
    nbytes, action, key, data = 0, None, None, b''
    while True:
        data_ = sock.recv(bufsize)
        if not data_:
            break
        if not data:
            nbytes, action, key, data = read_header(data_)
        else:
            data += data_
        if len(data) == nbytes:
            break
    assert nbytes == len(data), repr((nbytes, len(data), data))
    return action, key, data


class Server:
    """ Remote storage server
    """

    def __init__(self, host=None, port=None):
        if host is None:
            host = default_host
        if port is None:
            port = default_port
        self.host = host
        self.port = port

    def run(self, threaded=True, debug=False):
        """ Run server in a new thread.
        """
        import threading
        import socketserver

        class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):
            
            def handle(self, cache=defaultdict(dict)):
                try:
                    action, key, data = read_message(self.request)
                    if debug:
                        print("{} connected with action {!r} on {!r}"
                              .format(self.client_address[0], action, key))
                        print ('len(cache)=',len(cache))
                    dct = cache[key]
                    args, kwargs = pickle.loads(data)
                    result = getattr(dct, action)(*args, **kwargs)
                    if type(result).__name__ in ['dict_keys', 'dict_items', 'dict_values']:
                        result = tuple(result)
                    data = pickle.dumps(result)
                    message = make_message('return', digest(data), data)
                except Exception as exc:
                    tb = ('{:=^78}\n{}{:=^78}'
                          .format(' REMOTE HOST %s:%s '%(self.server.server_address),
                                  traceback.format_exc(), ''))
                    data = pickle.dumps((exc, tb))
                    message = make_message('raise', '', data)
                self.request.sendall(message)

        class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
            timeout = server_timeout
            allow_reuse_address = True

        class TCPServer(socketserver.TCPServer):
            timeout = server_timeout
            allow_reuse_address = True

        self.debug = debug

        if threaded:
            server = ThreadedTCPServer((self.host, self.port), ThreadedTCPRequestHandler)
            thread_name = 'remotedict-%s:%s'%(server.server_address)        
            self.server = server
            ip, port = server.server_address
            thread = threading.Thread(target=server.serve_forever,
                                      name=thread_name,
                                      daemon=True)
            self.thread = thread
            thread.start()
            if debug:
                print("Server loop running in thread:", thread.name)
        else:
            server = TCPServer((self.host, self.port), ThreadedTCPRequestHandler)
            thread_name = 'remotedict-%s:%s'%(server.server_address)
            self.server = server
            print("Server loop running in current thread:", thread_name)
            server.serve_forever()
            # TODO: show PID to kill
        return self

    def shutdown(self):
        """ Shutdown the server.
        """
        if self.debug:
            print('Shutting down server')
        self.server.shutdown()
        self.server.socket.shutdown(socket.SHUT_RDWR)
        self.server.socket.close()
        self.server = None
        
    def __del__(self):
        if self.server is not None:
            self.shutdown()

class Storage:
    """Frontend to named remote dictionary object.
    """

    def __init__(self, name=None, host=None, port=None):
        """
        Parameters
        ----------
        name : str
          Specify storage name
        host : str
          Specify server host name
        port : int
          Specify server port
        """
        if name is None:
            name = uuid.getnode()
        if host is None:
            host = default_host
        if port is None:
            port = default_port
        self.name = name
        self.host = host
        self.port = port

    def __apply_method(self, mthname, *args, **kwargs):
        return apply_method(mthname,
                            __key=self.name, __host=self.host, __port=self.port,
                            *args, **kwargs)

    def __repr__(self):
        return ('{}(name={!r}, host={!r}, port={})'
                .format(type(self).__name__, self.name, self.host, self.port))

    def __str__(self):
        return self.__apply_method('__str__')

    def __contains__(self, *args): return self.__apply_method('__contains__', *args)
    def __eq__(self, other): return self.__apply_method('__eq__', other)
    def __ne__(self, other): return self.__apply_method('__ne__', other)
    def __lt__(self, other): return self.__apply_method('__lt__', other)
    def __le__(self, other): return self.__apply_method('__le__', other)
    def __gt__(self, other): return self.__apply_method('__gt__', other)
    def __ge__(self, other): return self.__apply_method('__ge__', other)
    def __iter__(self, *args): return self.__apply_method('__iter__', *args)

    def __len__(self): return self.__apply_method('__len__')
    def __sizeof__(self): return self.__apply_method('__sizeof__')
    
    def __getitem__(self, key):
        return self.__apply_method('__getitem__', key)

    def __setitem__(self, key, value):
        return self.__apply_method('__setitem__', key, value)

    def get(self, *args): return self.__apply_method('get', *args)

    def pop(self, *args): return self.__apply_method('pop', *args)

    def popitem(self, *args): return self.__apply_method('popitem', *args)

    def update(self, *args, **kwargs): return self.__apply_method('update', *args, **kwargs)
    
    def clear(self): return self.__apply_method('clear')
    
    def keys(self): return self.__apply_method('keys')

    def items(self): return self.__apply_method('items')

    def values(self): return self.__apply_method('values')

    def copy(self): return self.__apply_method('copy')

    def setdefault(self, *args, **kwargs): return self.__apply_method('setdefault', *args, **kwargs)


def apply_method(mthname, *args, **kwargs):
    """Low-level function to call a dictionary method over socket.
    """
    key = kwargs.pop('__key', '')
    host = kwargs.pop('__host', default_host)
    port = kwargs.pop('__port', default_port)
    data = pickle.dumps((args, kwargs))
    message = make_message(mthname, key, data)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((host, port))
        sock.sendall(message)        
        action, key, data = read_message(sock)
        sock.close()
        if action == 'return':
            return pickle.loads(data)
        if action == 'raise':
            exc, tb = pickle.loads(data)
            if not isinstance(exc, KeyError):
                print(tb)
            raise exc
        raise NotImplementedError(repr(action))

def run_server():
    server = Server().run(threaded=False, debug=True)


def test(run_server=False):
    if run_server:
        server = Server().run(True)
    d, d1 = Storage(), {}
    d.clear()
    assert d.copy() == d1
    d['a'] = d1['a'] = 1
    assert d.keys() == tuple(d1.keys()), repr((d.keys(), d1.keys()))
    assert d.values() == tuple(d1.values())
    assert d.items() == tuple(d1.items())
    assert d.copy() == d1
    assert d['a'] == d1['a']
    assert d.get('b') == d1.get('b')
    d.update(c=3)
    d1.update(c=3)
    d[123] = d1[123] = 'abc'
    assert tuple(k for k in d) == tuple(k for k in d1)
    assert d.copy() == d1
    assert str(d) == str(d1)
    assert len(d) == len(d1)
    d.pop('a'); d1.pop('a')
    assert d.copy() == d1
    assert 'c' in d
    assert 'd' not in d
    assert d
    assert not not d

    try:
        d[123456]
        raise AssertionError('expected KeyError')
    except KeyError:
        pass
    except:
        raise

    if run_server:
        server.shutdown()

        try:
            print(d)
            raise AssertionError('expected ConnectionRefusedError')
        except ConnectionRefusedError:
            pass
        except:
            raise
   
if __name__ == '__main__':
    test()
    #run_server()
