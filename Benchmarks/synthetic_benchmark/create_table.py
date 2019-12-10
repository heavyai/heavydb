import os
import sys
import subprocess
import random
import datetime
import pymapd
from multiprocessing import Pool, cpu_count
from argparse import ArgumentParser


class Column:
    def __init__(self, column_name, sql_type, lower, upper, step=1):
        self.column_name = column_name
        assert sql_type.upper() in ["INT", "BIGINT"]
        self.sql_type = sql_type.upper()
        assert upper > lower
        self.lower_bound = lower
        self.upper_bound = upper
        assert step >= 1
        self.step = step
        if self.sql_type in ["INT"]:
            assert (
                self.upper_bound * step <= 2 ** 31
            ), "Generated values are larger than 32-bit signed integer."

    def generateEntry(self):
        if self.sql_type in ["INT", "BIGINT"]:
            return self.generateInt() * self.step
        else:
            assert False, "SQL type " + self.sql_type + " not supported yet"

    def generateInt(self):
        return int(random.randint(self.lower_bound, self.upper_bound))

    def createColumnDetailsString(self):
        """
        Returns the ColumnDetails as expected by pymapd's API
        """
        result = "ColumnDetails(name='"
        result += self.column_name
        result += "', type='"
        result += self.sql_type.upper()
        result += "', nullable=True, precision=0, scale=0, comp_param=0, encoding='NONE', is_array=False)"
        return result


class SyntheticTable:
    def __init__(self, **kwargs):
        """
            kwargs:
                table_name(str): synthetic table's name in the database
                fragment_size(int): fragment size (number of entries per fragment)
                num_fragment(int): total number of fragments for the synthetic table
                db_user(str): database username
                db_password(str): database password
                db_port(int): database port
                db_name(str): database name
                db_server(str): database server name
                data_dir_path(str): path to directory that will include the generated data
                is_remote_server(Bool): if True, it indicates that this class is not created on the 
                same machine that is going to host the server.  
        """
        self.table_name = kwargs["table_name"]
        self.fragment_size = kwargs["fragment_size"]
        self.num_fragments = kwargs["num_fragments"]
        self.db_name = kwargs["db_name"]
        self.db_user = kwargs["db_user"]
        self.db_password = kwargs["db_password"]
        self.db_server = kwargs["db_server"]
        self.db_port = kwargs["db_port"]
        self.data_dir_path = kwargs["data_dir_path"]
        self.num_entries = self.num_fragments * self.fragment_size
        self.column_list = self.generateColumnsSchema()
        self.data_dir_path = kwargs["data_dir_path"]
        self.is_remote_server = kwargs["is_remote_server"]
        if not os.path.isdir(self.data_dir_path):
            os.mkdir(self.data_dir_path)
        self.data_file_name_base = self.data_dir_path + "/data"

    def createDataAndImportTable(self, skip_data_generation=False):
        # deciding whether it is required to generate data and import it into the database
        # or the data already exists there:
        if (
            self.doesTableHasExpectedSchemaInDB()
            and self.doesTableHasExpectedNumEntriesInDB()
        ):
            print(
                "Data already exists in the database, proceeding to the queries:"
            )
        else:
            if self.is_remote_server:
                # at this point, we abort the procedure as the data is
                # either not present in the remote server or the schema/number of rows
                # does not match of those indicated by this class.
                raise Exception(
                    "Proper data does not exist in the remote server."
                )
            else:
                # generate random synthetic data
                if not skip_data_generation:
                    # choosing a relatively unique name for the generated csv files
                    current_time = str(datetime.datetime.now()).split()
                    self.data_file_name_base += "_" + current_time[0]

                    self.generateDataParallel()
                    print(
                        "Synthetic data created: "
                        + str(self.num_entries)
                        + " rows"
                    )
                # create a table on the database:
                self.createTableInDB()
                # import the generated data into the data base:
                self.importDataIntoTableInDB()
                print("Data imported into the database")

    def generateColumnsSchema(self):
        column_list = []
        # columns with uniform distribution and step=1
        column_list.append(Column("x10", "INT", 1, 10))
        column_list.append(Column("y10", "INT", 1, 10))
        column_list.append(Column("z10", "INT", 1, 10))
        column_list.append(Column("x100", "INT", 1, 100))
        column_list.append(Column("y100", "INT", 1, 100))
        column_list.append(Column("z100", "INT", 1, 100))
        column_list.append(Column("x1k", "INT", 1, 1000))
        column_list.append(Column("x10k", "INT", 1, 10000))
        column_list.append(Column("x100k", "INT", 1, 100000))
        column_list.append(Column("x1m", "INT", 1, 1000000))
        column_list.append(Column("x10m", "INT", 1, 10000000))

        # columns with step != 1
        # cardinality = 10k, range = 100m
        column_list.append(Column("x10k_s10k", "BIGINT", 1, 10000, 10000))
        column_list.append(Column("x100k_s10k", "BIGINT", 1, 100000, 10000))
        column_list.append(Column("x1m_s10k", "BIGINT", 1, 1000000, 10000))
        return column_list

    def getCreateTableCommand(self):
        create_sql = "CREATE TABLE " + self.table_name + " ( "
        for column_idx in range(len(self.column_list)):
            column = self.column_list[column_idx]
            create_sql += column.column_name + " " + column.sql_type
            if column_idx != (len(self.column_list) - 1):
                create_sql += ", "
        create_sql += ")"
        if self.fragment_size != 32000000:
            create_sql += (
                " WITH (FRAGMENT_SIZE = " + str(self.fragment_size) + ")"
            )
        create_sql += ";"
        return create_sql

    def getCopyFromCommand(self):
        copy_sql = "COPY " + self.table_name + " FROM '"
        copy_sql += (
            self.data_file_name_base + "*.csv' WITH (header = 'false');"
        )
        return copy_sql

    def generateData(self, thread_idx, size):
        """
            Single-thread random data generation based on the provided schema.
            Data is stored in CSV format.
        """
        file_name = (
            self.data_file_name_base + "_part" + str(thread_idx) + ".csv"
        )
        with open(file_name, "w") as f:
            for i in range(size):
                f.write(
                    ",".join(
                        map(
                            str,
                            [col.generateEntry() for col in self.column_list],
                        )
                    )
                )
                f.write("\n")

    def generateDataParallel(self):
        """
            Uses all available CPU threads to generate random data based on the 
            provided schema. Data is stored in CSV format.
        """
        num_threads = cpu_count()
        num_entries_per_thread = int(
            (self.num_entries + num_threads - 1) / num_threads
        )
        thread_index = [i for i in range(0, num_threads)]

        # making sure we end up having as many fragments as the user asked for
        num_balanced_entries = [
            num_entries_per_thread for _ in range(num_threads)
        ]
        if self.num_entries != num_entries_per_thread * num_threads:
            last_threads_portion = (
                self.num_entries - num_entries_per_thread * (num_threads - 1)
            )
            num_balanced_entries[-1] = last_threads_portion

        arguments = zip(thread_index, num_balanced_entries)

        with Pool(num_threads) as pool:
            pool.starmap(self.generateData, arguments)

    def createExpectedTableDetails(self):
        """
        Creates table details in the same format as expected 
        from pymapd's get_table_details  
        """
        return [
            column.createColumnDetailsString() for column in self.column_list
        ]

    def doesTableHasExpectedSchemaInDB(self):
        """
            Verifies whether the existing table in the database has the expected
            schema or not. 
        """
        try:
            con = pymapd.connect(
                user=self.db_user,
                password=self.db_password,
                host=self.db_server,
                port=self.db_port,
                dbname=self.db_name,
            )
        except:
            raise Exception("Pymapd's connection to the server has failed.")
        try:
            table_details = con.get_table_details(self.table_name)
        except:
            # table does not exist
            print("Table does not exist in the database")
            return False

        if [
            str(table_detail) for table_detail in table_details
        ] == self.createExpectedTableDetails():
            return True
        else:
            print("Schema does not match the expected one:")
            print(
                "Observed table details: "
                + str([str(table_detail) for table_detail in table_details])
            )
            print(
                "Expected table details: "
                + str(self.createExpectedTableDetails())
            )

    def doesTableHasExpectedNumEntriesInDB(self):
        """
            Verifies whether the existing table in the database has the expected
            number of entries in it as in this class.
        """
        try:
            con = pymapd.connect(
                user=self.db_user,
                password=self.db_password,
                host=self.db_server,
                port=self.db_port,
                dbname=self.db_name,
            )
            result = con.execute(
                "select count(*) from " + self.table_name + ";"
            )
            if list(result)[0][0] == self.num_entries:
                return True
            else:
                print("Expected num rows did not match:")
                return False
        except:
            raise Exception("Pymapd's connection to the server has failed.")

    def createTableInDB(self):
        try:
            con = pymapd.connect(
                user=self.db_user,
                password=self.db_password,
                host=self.db_server,
                port=self.db_port,
                dbname=self.db_name,
            )
            # drop the current table if exists:
            con.execute("DROP TABLE IF EXISTS " + self.table_name + ";")
            # create a new table:
            con.execute(self.getCreateTableCommand())
        except:
            raise Exception("Failure in creating a new table.")

    def importDataIntoTableInDB(self):
        try:
            con = pymapd.connect(
                user=self.db_user,
                password=self.db_password,
                host=self.db_server,
                port=self.db_port,
                dbname=self.db_name,
            )
            # import generated data:
            con.execute(self.getCopyFromCommand())
        except:
            raise Exception("Failure in importing data into the table")


if __name__ == "__main__":
    parser = ArgumentParser()
    required = parser.add_argument_group("required arguments")
    required.add_argument("--user", dest="user", default="admin")
    required.add_argument(
        "--password", dest="password", default="HyperInteractive"
    )
    required.add_argument(
        "--table_name", dest="table_name", default="synthetic_test_table"
    )
    required.add_argument("--fragment_size", dest="fragment_size", default="1")
    required.add_argument(
        "--num_fragments", dest="num_fragments", default="128"
    )
    required.add_argument("--name", dest="name", default="omnisci")
    required.add_argument("--server", dest="server", default="localhost")
    required.add_argument("--port", dest="port", default="6274")
    required.add_argument(
        "--data_dir",
        dest="data_dir",
        default=os.getcwd() + "/../build/synthetic_data",
    )
    required.add_argument(
        "--just_data_generation",
        dest="just_data_generation",
        action="store_true",
        help="Indicates that the code will only generates synthetic data, bypassing all "
        + "other capabilities. The generated data will be stored in DATA_DIR.",
    )
    required.add_argument(
        "--just_data_import",
        dest="just_data_import",
        action="store_true",
        help="Indicates that the code assumes the data is generated and exists at data_dir/*.csv. "
        + "It then proceeds with table creation and data import. ",
    )
    args = parser.parse_args()

    synthetic_table = SyntheticTable(
        table_name=args.table_name,
        fragment_size=int(args.fragment_size),
        num_fragments=int(args.num_fragments),
        db_name=args.name,
        db_user=args.user,
        db_password=args.password,
        db_server=args.server,
        db_port=int(args.port),
        data_dir_path=args.data_dir,
        is_remote_server=False,
    )

    if (args.just_data_generation is False) and (
        args.just_data_import is False
    ):
        synthetic_table.createDataAndImportTable()
    elif args.just_data_generation is True:
        synthetic_table.generateDataParallel()
        print(
            "Synthetic data created: "
            + str(synthetic_table.num_entries)
            + " rows"
        )
    else:
        synthetic_table.createDataAndImportTable(args.just_data_import)
