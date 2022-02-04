import argparse
import sys
import pymapd

from omnisci.thrift.ttypes import TDashboard

def getOptions(args=None):
    parser = argparse.ArgumentParser(description='Basic benchmark for system tables')
    parser.add_argument('--host', help='OmniSci server address', default='localhost')
    parser.add_argument('--port', help='OmniSci server port', default='6273')
    parser.add_argument('--user', help='OmniSci user name', default='admin')
    parser.add_argument('--password', help='OmniSci password', default='HyperInteractive')
    parser.add_argument('--database_count', help='Number of databases to create', default=1)
    parser.add_argument('--table_count', help='Number of tables to create', default=100)
    parser.add_argument('--dashboard_count', help='Number of dashboards to create', default=100)
    parser.add_argument('--user_count', help='Number of users to create', default=10)
    parser.add_argument('--role_count', help='Number of roles to create', default=5)
    parser.add_argument('--skip_object_creation', help='Skip creation of database objects', default=False)
    parser.add_argument('--skip_object_deletion', help='Skip deletion of database objects', default=False)
    parser.add_argument('--tag', help='Tag for test run')
    return parser.parse_args(args)

class OmniSciCon:
    def __init__(self, user, password, db_name, host):
        self.con = pymapd.connect(user=user, password=password, dbname=db_name, host=host)
        self.cursor = self.con.cursor()

    def query(self, sql):
        return self.cursor.execute(sql)

    def create_dashboard(self, dashboard_name):
        dashboard = TDashboard(dashboard_name = dashboard_name)
        return self.con.create_dashboard(dashboard)

def create_database(omnisci_con, db_id):
    omnisci_con.query(f"CREATE DATABASE test_db_{db_id}")

def create_insert_and_select_from_table(omnisci_con, table_id):
    omnisci_con.query(f"CREATE TABLE test_table_{table_id} (a INTEGER, b TEXT)")
    for i in range(10):
        omnisci_con.query(f"INSERT INTO test_table_{table_id} VALUES ({i}, 'abc_{i}')")
    omnisci_con.query(f"SELECT AVG(a) FROM test_table_{table_id}")

def create_dashboard(omnisci_con, dashboard_id):
    omnisci_con.create_dashboard(f"test_dashboard_{dashboard_id}")

def create_user(omnisci_con, user_id):
    omnisci_con.query(f"CREATE USER test_user_{user_id} (password = 'test_pass')")

def create_role(omnisci_con, role_id):
    omnisci_con.query(f"CREATE ROLE test_role_{role_id}")

def assign_role(omnisci_con, user_id, role_id):
    omnisci_con.query(f"GRANT test_role_{role_id} TO test_user_{user_id}")

def grant_role_table_select(omnisci_con, role_id, db_id):
    omnisci_con.query(f"GRANT SELECT ON DATABASE test_db_{db_id} TO test_role_{role_id}")

def grant_user_table_select(omnisci_con, user_id, db_id):
    omnisci_con.query(f"GRANT SELECT ON DATABASE test_db_{db_id} TO test_user_{user_id}")

def drop_database(omnisci_con, db_id):
    omnisci_con.query(f"DROP DATABASE test_db_{db_id}")

def drop_user(omnisci_con, user_id):
    omnisci_con.query(f"DROP USER test_user_{user_id}")

def drop_role(omnisci_con, role_id):
    omnisci_con.query(f"DROP ROLE test_role_{role_id}")

def query_and_time_system_table(omnisci_con, table_name):
    query = f"SELECT COUNT(*) FROM {table_name}"
    result = omnisci_con.query(query)
    print(f"Query: {query}, Execution time: {result._result.execution_time_ms}ms")
    query = f"SELECT * FROM {table_name} LIMIT 10"
    result = omnisci_con.query(query)
    print(f"Query: {query}, Execution time: {result._result.execution_time_ms}ms")

def get_connection(options, db_name):
    return OmniSciCon(options.user, options.password, db_name, options.host)

def main(argv):
    options = getOptions(argv)
    default_db = "omnisci"
    omnisci_con = get_connection(options, default_db)

    if not options.skip_object_creation:
        print("Creating database objects")
        for db_id in range(options.database_count):
            create_database(omnisci_con, db_id)
            db_name = f"test_db_{db_id}"
            omnisci_con = get_connection(options, db_name)
            for table_id in range(options.table_count):
                create_insert_and_select_from_table(omnisci_con, table_id)
            print(f"{options.table_count} tables created for {db_name}")
            for dashboard_id in range(options.dashboard_count):
                create_dashboard(omnisci_con, dashboard_id)
            print(f"{options.dashboard_count} dashboards created for {db_name}")
        print(f"{options.database_count} databases created")

        omnisci_con = get_connection(options, default_db)
        for user_id in range(options.user_count):
            create_user(omnisci_con, user_id)
        print(f"{options.user_count} users created")

        for role_id in range(options.role_count):
            create_role(omnisci_con, role_id)
        print(f"{options.role_count} roles created")

        half_roles = int(options.role_count / 2)
        for user_id in range(options.user_count):
            for role_id in range(half_roles):
                assign_role(omnisci_con, user_id, role_id)

        if options.database_count > 0:
            db_id = 0
            for role_id in range(half_roles):
                grant_role_table_select(omnisci_con, role_id + half_roles, db_id)

            half_users = int(options.user_count / 2)
            for user_id in range(half_users):
                grant_user_table_select(omnisci_con, user_id + half_users, db_id)

    system_tables = ["tables",
                     "dashboards",
                     "databases",
                     "users",
                     "permissions",
                     "role_assignments",
                     "roles",
                     "storage_details",
                     "memory_details",
                     "memory_summary"]
    omnisci_con = get_connection(options, "information_schema")
    print("Executing system table queries")
    for table_name in system_tables:
        query_and_time_system_table(omnisci_con, table_name)

    if not options.skip_object_deletion:
        omnisci_con = get_connection(options, default_db)
        print("Dropping databases")
        for db_id in range(options.database_count):
            drop_database(omnisci_con, db_id)

        print("Dropping users")
        for user_id in range(options.user_count):
            drop_user(omnisci_con, user_id)

        print("Dropping roles")
        for role_id in range(options.role_count):
            drop_role(omnisci_con, role_id)

if __name__ == "__main__":
    main(sys.argv[1:])
