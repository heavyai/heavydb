import sys
import pymapd
import pyarrow as pa
import pandas as pd
import numpy as np
import time

import argparse

def getOptions(args=None):
    parser = argparse.ArgumentParser(description='Benchmark OmniSci batch and streaming table ingest')
    parser.add_argument('-s','--host', help='OmniSci server address', default='localhost')
    parser.add_argument('-p','--port', help='OmniSci server port', default='6273')
    parser.add_argument('-d','--db', help='OmniSci database name', default='omnisci')
    parser.add_argument('-u','--user', help='OmniSci user name', default='admin')
    parser.add_argument('-w','--password', help='OmniSci password', default='HyperInteractive')
    parser.add_argument('-e','--max_rollback_epochs', help='Max Rollback Epochs', type=int, default=-1)
    parser.add_argument('-t','--temp_table', help='Use temporary table', type=bool, default=False)
    parser.add_argument('-r','--num_rows', help='Number of rows to benchmark with', type=int, default=10000)
    return parser.parse_args(args)


class OmniCon:
    def __init__(self, user, pw, dbname):
        self.con = pymapd.connect(user=user, password=pw, dbname=dbname, host="localhost")
        self.cursor = self.con.cursor()

    def query(self, sql):
        return self.cursor.execute(sql)

def create_table(omni_con, table_name, is_temporary=False, max_rollback_epochs=-1):
    drop_sql = "DROP TABLE IF EXISTS " + table_name
    optional_temp_stmt = "TEMPORARY" if is_temporary else ""
    optional_max_rollback_stmt = "WITH (max_rollback_epochs={max_rollback_epochs})".format(max_rollback_epochs=max_rollback_epochs) if max_rollback_epochs >= 0 else ""
    create_sql = "CREATE {optional_temp_stmt} TABLE {table_name} (a INTEGER, b INTEGER, c INTEGER, d INTEGER) {optional_max_rollback_stmt}".format(optional_temp_stmt = optional_temp_stmt, table_name=table_name, optional_max_rollback_stmt=optional_max_rollback_stmt)
    omni_con.query(drop_sql)
    omni_con.query(create_sql)

def gen_data(num_rows):
    df = pd.DataFrame(np.random.randint(0,100,size=(num_rows, 4)), columns=['a','b','c','d'])
    df = df.astype(np.int32)
    return df

def bench_streaming_sql_inserts(omni_con, table_name, data):
    num_rows = len(data.index)
    base_insert_sql = "INSERT INTO " + table_name + "(a, b, c, d) VALUES ({0}, {1}, {2}, {3})"
    insert_statements = []
    for r in range(num_rows):
        insert_statements.append(base_insert_sql.format(data.iat[r,0], data.iat[r,1], data.iat[r,2], data.iat[r,3]))
    start_time = time.perf_counter()
    for r in range(num_rows):
        omni_con.query(insert_statements[r])
    end_time = time.perf_counter()
    time_diff = end_time - start_time
    rows_per_second = num_rows / time_diff
    print("Streaming – SQL Inserts: {0} rows in {1} seconds at {2} rows/sec".format(num_rows, time_diff, rows_per_second))

def bench_bulk_columnar(omni_con, table_name, data):
    num_rows = len(data.index)
    start_time = time.perf_counter()
    omni_con.con.load_table_columnar(table_name, data, preserve_index=False)
    end_time = time.perf_counter()
    time_diff = end_time - start_time
    rows_per_second = num_rows / time_diff
    print("Bulk load – Columnar: {0} rows in {1} seconds at {2} rows/sec".format(num_rows, time_diff, rows_per_second))

def bench_bulk_arrow(omni_con, table_name, data):
    num_rows = len(data.index)
    arrow_data = pa.Table.from_pandas(data)
    start_time = time.perf_counter()
    omni_con.con.load_table_arrow(table_name, arrow_data, preserve_index=False)
    end_time = time.perf_counter()
    time_diff = end_time - start_time
    rows_per_second = num_rows / time_diff
    print("Bulk load – Arrow: {0} rows in {1} seconds at {2} rows/sec".format(num_rows, time_diff, rows_per_second))

def bench_streaming_columnar(omni_con, table_name, data):
    num_rows = len(data.index)
    start_time = time.perf_counter()
    for r in range(num_rows):
        row_df = data.iloc[r:r+1]
        omni_con.con.load_table_columnar(table_name, row_df, preserve_index=False)
    end_time = time.perf_counter()
    time_diff = end_time - start_time
    rows_per_second = num_rows / time_diff
    print("Streaming – Columnar: {0} rows in {1} seconds at {2} rows/sec".format(num_rows, time_diff, rows_per_second))

def main(argv):
    options = getOptions(argv)
    omni_con = OmniCon(options.user, options.password, options.db) 

    data = gen_data(options.num_rows)

    table_name = "stream_insert_sql"
    create_table(omni_con, table_name, options.temp_table, options.max_rollback_epochs)
    bench_streaming_sql_inserts(omni_con, table_name, data)

    #Below is too slow to bench at any real scale 
    #table_name = "stream_columnar"
    #create_table(omni_con, table_name, options.temp_table, options.max_rollback_epochs)
    #bench_streaming_columnar(omni_con, table_name, data)

    table_name = "bulk_columnar"
    create_table(omni_con, table_name, options.temp_table, options.max_rollback_epochs)
    bench_bulk_columnar(omni_con, table_name, data)

    table_name = "bulk_arrow"
    create_table(omni_con, table_name, options.temp_table, options.max_rollback_epochs)
    bench_bulk_arrow(omni_con, table_name, data)

if __name__ == "__main__":
    main(sys.argv[1:])