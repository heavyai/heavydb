#!/usr/bin/env python

from thrift.protocol import TBinaryProtocol
from thrift.protocol import TJSONProtocol
from thrift.transport import TSocket
from thrift.transport import THttpClient
from thrift.transport import TTransport
from mapd import MapD
from mapd import ttypes
import time


def get_client(host_or_uri, port, http):
  if http:
    transport = THttpClient.THttpClient(host_or_uri)
    protocol = TJSONProtocol.TJSONProtocol(transport)
  else:
    socket = TSocket.TSocket(host_or_uri, port)
    transport = TTransport.TBufferedTransport(socket)
    protocol = TBinaryProtocol.TBinaryProtocol(transport)

  client = MapD.Client(protocol)
  transport.open()
  return client

def main():

    db_name = 'mapd'
    user_name = 'mapd'
    passwd = 'HyperInteractive'
    hostname = 'test.mapd.com'
    portno = 9091

    client = get_client("localhost",portno,False)
    session = client.connect(user_name, passwd, db_name)
    print 'Connection complete'
    query = 'select a,b from table1 limit 25;'
    print 'Query is : ' + query

    # always use True for is columnar
    results = client.sql_execute(session, query, True, None)
    dates = ['TIME', 'TIMESTAMP', 'DATE']

    if results.row_set.is_columnar == True:
        number_of_rows = list(range(0, len(results.row_set.columns[0].nulls)))
        number_of_columns = list(range(0, len(results.row_set.row_desc)))
        for n in number_of_rows:
            for i in number_of_columns:
                column_type = ttypes.TDatumType._VALUES_TO_NAMES[results.row_set.row_desc[i].col_type.type]
                column_name = results.row_set.row_desc[i].col_name
                column_array = results.row_set.row_desc[i].col_type.is_array
                if not column_array:
                    if column_type in ['SMALLINT', 'INT', 'BIGINT', 'TIME', 'TIMESTAMP', 'DATE', 'BOOL']:
                        column_value = results.row_set.columns[i].data.int_col[n]
                    if column_type in ['FLOAT', 'DECIMAL', 'DOUBLE']:
                        column_value = results.row_set.columns[i].data.real_col[n]
                    if column_type in ['STR']:
                        column_value = results.row_set.columns[i].data.str_col[n]
                else:
                    column_value = results.row_set.columns[i].data.arr_col[n].data.str_col
                print(column_name)
                if column_type in dates:
                    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(column_value)))
                else:
                    print(column_value)

    else:
        print('Please use columns not rows in query execution')
        client.disconnect(session)
        quit()
        
    client.disconnect(session)


if __name__ == '__main__':
    main()
