#!/usr/bin/env python

from thrift.protocol import TBinaryProtocol
from thrift.transport import TSocket
from thrift.transport import TTransport

from mapd import MapD


def get_client(host, port):
  socket = TSocket.TSocket(host, port)
  transport = TTransport.TBufferedTransport(socket)
  protocol = TBinaryProtocol.TBinaryProtocol(transport)
  client = MapD.Client(protocol)
  transport.open()
  return client


def import_some_rows(session, table_name, client):
  input_rows = []
  for i in xrange(0, 100):
    input_row = MapD.TStringRow()
    input_row.cols = [
      MapD.TStringValue(str(i)),
      MapD.TStringValue('str%d' % i),
      MapD.TStringValue('real_str%d' % i)
    ]
    input_rows.append(input_row)
  client.load_table(session, table_name, input_rows)


def main():
  # Import rows to a table created by the following statement:
  # CREATE TABLE foo(x int, str text encoding dict, real_str text);

  # How to run:
  #
  # thrift -gen py ~/mapd2/mapd.thrift
  # PYTHONPATH=`pwd`/gen-py python ./stream_insert.py

  table_name = 'foo'
  db_name = 'mapd'
  user_name = 'mapd'
  passwd = 'HyperInteractive'
  hostname = 'localhost'
  portno = 9091

  client = get_client(hostname, portno)
  session = client.connect(user_name, passwd, db_name)
  import_some_rows(session, table_name, client)
  client.disconnect(session)



if __name__ == "__main__":
  main()
