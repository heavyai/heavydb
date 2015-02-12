#!/usr/bin/env python

import readline
import logging
import sys

from thrift.transport import TTransport
from thrift.transport import TSocket
from thrift.transport import TSSLSocket
from thrift.transport import THttpClient
from thrift.protocol import TBinaryProtocol

from mapd import MapD
from mapd.ttypes import *

LOG_FILENAME = '/tmp/mapd_completer.log'
logging.basicConfig(filename=LOG_FILENAME,
                    level=logging.DEBUG,
                    )


class SimpleCompleter(object):

    def __init__(self, options):
        self.options = sorted(options)
        return

    def complete(self, text, state):
        response = None
        if state == 0:
            # This is the first time for this text, so build a match list.
            if text:
                self.matches = [s
                                for s in self.options
                                if s and s.startswith(text)]
                logging.debug('%s matches: %s', repr(text), self.matches)
            else:
                self.matches = self.options[:]
                logging.debug('(empty input) matches: %s', self.matches)

        # Return the state'th item from the match list,
        # if we have that many.
        try:
            response = self.matches[state]
        except IndexError:
            response = None
        logging.debug('complete(%s, %s) => %s',
                      repr(text), state, repr(response))
        return response


def str_row(row):
    return '|'.join([
        str(datum.int_val) if datum.type == TDatumType.INT else str(datum.real_val) for datum in row ])


def input_loop(client):
    while True:
        try:
            line = raw_input('mapd> ')
            if line == 'quit':
                return
            query_result = client.select(line)
        except InvalidQueryException as ex:
            print ex.error_msg
            continue
        for row in query_result.rows:
            print str_row(row)


if len(sys.argv) <= 1 or sys.argv[1] == '--help':
  print('')
  print('Usage: ' + sys.argv[0] + ' [-h host[:port]]')
  print('')
  sys.exit(0)

host = 'localhost'
port = 9090
argi = 1

if sys.argv[argi] == '-h':
  parts = sys.argv[argi+1].split(':')
  host = parts[0]
  if len(parts) > 1:
    port = int(parts[1])
  argi += 2

socket = TSocket.TSocket(host, port)
transport = TTransport.TBufferedTransport(socket)
protocol = TBinaryProtocol.TBinaryProtocol(transport)
client = MapD.Client(protocol)
transport.open()

# Register our completer function
readline.set_completer(SimpleCompleter(['SELECT', 'FROM', 'WHERE', 'CREATE', 'TABLE']).complete)

# Use the tab key for completion
readline.parse_and_bind('tab: complete')

# Prompt the user for text
input_loop(client)
