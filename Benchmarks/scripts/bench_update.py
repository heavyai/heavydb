import argparse
import sys
import pymapd
import pandas as pd
import numpy as np

from pathlib import Path

def getOptions(args=None):
    parser = argparse.ArgumentParser(description='Basic benchmark for update queries')
    parser.add_argument('-s','--host', help='OmniSci server address', default='localhost')
    parser.add_argument('-p','--port', help='OmniSci server port', default='6273')
    parser.add_argument('-d','--db', help='OmniSci database name', default='omnisci')
    parser.add_argument('-u','--user', help='OmniSci user name', default='admin')
    parser.add_argument('-w','--password', help='OmniSci password', default='HyperInteractive')
    parser.add_argument('-r', '--num_rows', help='Number of rows to benchmark with', type=int, default=1_000_000)
    parser.add_argument('-t', '--tag', help='Tag for test run')
    return parser.parse_args(args)

class OmniSciCon:
    def __init__(self, user, pw, dbname, host):
        self.con = pymapd.connect(user=user, password=pw, dbname=dbname, host=host)
        self.cursor = self.con.cursor()

    def query(self, sql):
        return self.cursor.execute(sql)

def create_and_import_into_table(omnisci_con, table_name, file_path):
    drop_sql = f"DROP TABLE IF EXISTS {table_name}"
    create_sql = f"CREATE TABLE {table_name} (a INTEGER, b INTEGER, c INTEGER, d DOUBLE)"
    import_sql = f"COPY {table_name} FROM '{file_path.absolute()}'"
    omnisci_con.query(drop_sql)
    omnisci_con.query(create_sql)
    omnisci_con.query(import_sql)

def gen_data(num_rows):
    df = pd.DataFrame(np.random.randint(0, num_rows, size=(num_rows, 4)), columns=['a', 'b', 'c', 'd'])
    df = df.astype(np.int32)
    return df

def bench_update_query(omnisci_con, table_name, tag):
    query = f"UPDATE {table_name} SET a = a + 10, c = c + b, d = d * 2 WHERE MOD(b, 2) = 0"
    query_times = []
    for i in range(10):
        result = omnisci_con.query(query)
        query_times.append(result._result.execution_time_ms)

    print(f"Test tag: {tag}\nQuery: {query}\nRaw times(ms): {query_times}\n"
          f"Avg: {np.average(query_times)}\nMin: {np.min(query_times)}\n"
          f"Max: {np.max(query_times)}\nMedian: {np.median(query_times)}")

def main(argv):
    options = getOptions(argv)
    omnisci_con = OmniSciCon(options.user, options.password, options.db, options.host)

    file_path = Path(f"{options.num_rows}_rand.csv")
    if not file_path.exists():
        df = gen_data(options.num_rows)
        df.to_csv(file_path, index=False)

    table_name = "update_bench_test"
    create_and_import_into_table(omnisci_con, table_name, file_path)

    bench_update_query(omnisci_con, table_name, options.tag)

if __name__ == "__main__":
    main(sys.argv[1:])
