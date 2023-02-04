import argparse
import sys
import pymapd
import pandas as pd
import numpy as np

from pathlib import Path

def getOptions(args=None):
    parser = argparse.ArgumentParser(description='Basic benchmark for update queries')
    parser.add_argument('-s','--host', help='HEAVY.AI server address', default='localhost')
    parser.add_argument('-p','--port', help='HEAVY.AI server port', default='6273')
    parser.add_argument('-d','--db', help='HEAVY.AI database name', default='heavyai')
    parser.add_argument('-u','--user', help='HEAVY.AI user name', default='admin')
    parser.add_argument('-w','--password', help='HEAVY.AI password', default='HyperInteractive')
    parser.add_argument('-r', '--num_rows', help='Number of rows to benchmark with', type=int, default=1_000_000)
    parser.add_argument('-t', '--tag', help='Tag for test run')
    return parser.parse_args(args)

class HeavyAICon:
    def __init__(self, user, pw, dbname, host):
        self.con = pymapd.connect(user=user, password=pw, dbname=dbname, host=host)
        self.cursor = self.con.cursor()

    def query(self, sql):
        return self.cursor.execute(sql)

def create_and_import_into_table(heavyai_con, table_name, file_path):
    drop_sql = f"DROP TABLE IF EXISTS {table_name}"
    create_sql = f"CREATE TABLE {table_name} (a INTEGER, b INTEGER, c INTEGER, d DOUBLE)"
    import_sql = f"COPY {table_name} FROM '{file_path.absolute()}'"
    heavyai_con.query(drop_sql)
    heavyai_con.query(create_sql)
    heavyai_con.query(import_sql)

def gen_data(num_rows):
    df = pd.DataFrame(np.random.randint(0, num_rows, size=(num_rows, 4)), columns=['a', 'b', 'c', 'd'])
    df = df.astype(np.int32)
    return df

def bench_update_query(heavyai_con, table_name, tag):
    query = f"UPDATE {table_name} SET a = a + 10, c = c + b, d = d * 2 WHERE MOD(b, 2) = 0"
    query_times = []
    for i in range(10):
        result = heavyai_con.query(query)
        query_times.append(result._result.execution_time_ms)

    print(f"Test tag: {tag}\nQuery: {query}\nRaw times(ms): {query_times}\n"
          f"Avg: {np.average(query_times)}\nMin: {np.min(query_times)}\n"
          f"Max: {np.max(query_times)}\nMedian: {np.median(query_times)}")

def main(argv):
    options = getOptions(argv)
    heavyai_con = HeavyAICon(options.user, options.password, options.db, options.host)

    file_path = Path(f"{options.num_rows}_rand.csv")
    if not file_path.exists():
        df = gen_data(options.num_rows)
        df.to_csv(file_path, index=False)

    table_name = "update_bench_test"
    create_and_import_into_table(heavyai_con, table_name, file_path)

    bench_update_query(heavyai_con, table_name, options.tag)

if __name__ == "__main__":
    main(sys.argv[1:])
