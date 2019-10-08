import os
import timeit
import logging
import uuid
import datetime
import json
import pymapd
import pandas
import re
from argparse import ArgumentParser

# For usage info, run: `./<script_name>.py --help`


def get_connection(**kwargs):
    """
      Connects to the db using pymapd
      https://pymapd.readthedocs.io/en/latest/usage.html#connecting

      Kwargs:
        db_user(str): DB username
        db_passwd(str): DB password
        db_server(str): DB host
        db_port(int): DB port
        db_name(str): DB name

      Returns:
        con(class): Connection class
        False(bool): The connection failed. Exception should be logged.
    """
    try:
        logging.debug("Connecting to mapd db...")
        con = pymapd.connect(
            user=kwargs["db_user"],
            password=kwargs["db_passwd"],
            host=kwargs["db_server"],
            port=kwargs["db_port"],
            dbname=kwargs["db_name"],
        )
        logging.info("Succesfully connected to mapd db")
        return con
    except (pymapd.exceptions.OperationalError, pymapd.exceptions.Error):
        logging.exception("Error connecting to database.")
        return False


def json_format_handler(x):
    # Function to allow json to deal with datetime and numpy int
    if isinstance(x, datetime.datetime):
        return x.isoformat()
    raise TypeError("Unknown type")


# Parse input parameters
parser = ArgumentParser()
optional = parser._action_groups.pop()
required = parser.add_argument_group("required arguments")
parser._action_groups.append(optional)
optional.add_argument(
    "-v", "--verbose", action="store_true", help="Turn on debug logging"
)
optional.add_argument(
    "-q",
    "--quiet",
    action="store_true",
    help="Suppress script outuput " + "(except warnings and errors)",
)
required.add_argument(
    "-u", "--user", dest="user", default="mapd", help="Source database user"
)
required.add_argument(
    "-p",
    "--passwd",
    dest="passwd",
    default="HyperInteractive",
    help="Source database password",
)
required.add_argument(
    "-s",
    "--server",
    dest="server",
    default="localhost",
    help="Source database server hostname",
)
optional.add_argument(
    "-o",
    "--port",
    dest="port",
    type=int,
    default=6274,
    help="Source database server port",
)
required.add_argument(
    "-n", "--name", dest="name", default="mapd", help="Source database name"
)
required.add_argument(
    "-l", "--label", dest="label", required=True, help="Benchmark run label"
)
required.add_argument(
    "-f",
    "--import-file",
    dest="import_file",
    required=True,
    help="Absolute path to file on omnisci_server machine with data for "
    + "import test",
)
required.add_argument(
    "-c",
    "--table-schema-file",
    dest="table_schema_file",
    required=True,
    help="Path to local file with CREATE TABLE sql statement for the "
    + "import table",
)
optional.add_argument(
    "-t",
    "--import-table-name",
    dest="import_table_name",
    default="import_benchmark_test",
    help="Name of table to import data to. NOTE: This table will be dropped "
    + "before and after the import test, unless "
    + "--no-drop-table-[before/after] is specified.",
)
optional.add_argument(
    "-F",
    "--import-query-template-file",
    dest="import_query_template_file",
    help="Path to file containing template for import query. "
    + 'The script will replace "##TAB##" with the value of import_table_name '
    + 'and "##FILE##" with the value of table_schema_file. By default, the '
    + "script will use the COPY FROM command with the default default "
    + "delimiter (,).",
)
optional.add_argument(
    "--no-drop-table-before",
    dest="no_drop_table_before",
    action="store_true",
    help="Do not drop the import table and recreate it before import "
    + "NOTE: Make sure existing table schema matches import .csv file schema",
)
optional.add_argument(
    "--no-drop-table-after",
    dest="no_drop_table_after",
    action="store_true",
    help="Do not drop the import table after import",
)
optional.add_argument(
    "-A",
    "--import-test-name",
    dest="import_test_name",
    help='Name of import test (ex: "ips"). Required when using '
    + "jenkins_bench_json as output.",
)
optional.add_argument(
    "-m", "--machine-name", dest="machine_name", help="Name of source machine"
)
optional.add_argument(
    "-a",
    "--machine-uname",
    dest="machine_uname",
    help="Uname info from " + "source machine",
)
optional.add_argument(
    "-e",
    "--destination",
    dest="destination",
    default="mapd_db",
    help="Destination type: [mapd_db, file_json, output, jenkins_bench] "
    + "Multiple values can be input seperated by commas, "
    + 'ex: "mapd_db,file_json"',
)
optional.add_argument(
    "-U",
    "--dest-user",
    dest="dest_user",
    default="mapd",
    help="Destination mapd_db database user",
)
optional.add_argument(
    "-P",
    "--dest-passwd",
    dest="dest_passwd",
    default="HyperInteractive",
    help="Destination mapd_db database password",
)
optional.add_argument(
    "-S",
    "--dest-server",
    dest="dest_server",
    help="Destination mapd_db database server hostname"
    + ' (required if destination = "mapd_db")',
)
optional.add_argument(
    "-O",
    "--dest-port",
    dest="dest_port",
    type=int,
    default=6274,
    help="Destination mapd_db database server port",
)
optional.add_argument(
    "-N",
    "--dest-name",
    dest="dest_name",
    default="mapd",
    help="Destination mapd_db database name",
)
optional.add_argument(
    "-T",
    "--dest-table",
    dest="dest_table",
    default="import_results",
    help="Destination mapd_db table name",
)
optional.add_argument(
    "-C",
    "--dest-table-schema-file",
    dest="dest_table_schema_file",
    default="results_table_schemas/import-results.sql",
    help="Destination table schema file. This must be an executable CREATE "
    + "TABLE statement that matches the output of this script. It is "
    + "required when creating the results table. Default location is in "
    + '"./results_table_schemas/query-results.sql"',
)
optional.add_argument(
    "-j",
    "--output-file-json",
    dest="output_file_json",
    help="Absolute path of .json output file "
    + '(required if destination = "file_json")',
)
optional.add_argument(
    "-J",
    "--output-file-jenkins",
    dest="output_file_jenkins",
    help="Absolute path of jenkins benchmark .json output file "
    + '(required if destination = "jenkins_bench")',
)
args = parser.parse_args()
if args.verbose:
    logging.basicConfig(level=logging.DEBUG)
elif args.quiet:
    logging.basicConfig(level=logging.WARNING)
else:
    logging.basicConfig(level=logging.INFO)
source_db_user = args.user
source_db_passwd = args.passwd
source_db_server = args.server
source_db_port = args.port
source_db_name = args.name
label = args.label
import_file = args.import_file
table_schema_file = args.table_schema_file
import_table_name = args.import_table_name
import_query_template_file = args.import_query_template_file
no_drop_table_before = args.no_drop_table_before
no_drop_table_after = args.no_drop_table_after
import_test_name = args.import_test_name
machine_name = args.machine_name
machine_uname = args.machine_uname
destinations = args.destination.split(",")
if "mapd_db" in destinations:
    valid_destination_set = True
    dest_db_user = args.dest_user
    dest_db_passwd = args.dest_passwd
    if args.dest_server is None:
        # If dest_server is not set for mapd_db, then exit
        logging.error('"dest_server" is required when destination = "mapd_db"')
        exit(1)
    else:
        dest_db_server = args.dest_server
    dest_db_port = args.dest_port
    dest_db_name = args.dest_name
    dest_table = args.dest_table
    dest_table_schema_file = args.dest_table_schema_file
if "file_json" in destinations:
    valid_destination_set = True
    if args.output_file_json is None:
        # If output_file_json is not set for file_json, then exit
        logging.error(
            '"output_file_json" is required when destination = "file_json"'
        )
        exit(1)
    else:
        output_file_json = args.output_file_json
if "output" in destinations:
    valid_destination_set = True
if "jenkins_bench" in destinations:
    valid_destination_set = True
    if args.output_file_jenkins is None:
        # If output_file_jenkins is not set for jenkins_bench, then exit
        logging.error(
            '"output_file_jenkins" is required '
            + 'when destination = "jenkins_bench"'
        )
        exit(1)
    elif args.import_test_name is None:
        # If import_test_name is not set for jenkins_bench, then exit
        logging.error(
            '"import_test_name" is required '
            + 'when destination = "jenkins_bench"'
        )
        exit(1)
    else:
        output_file_jenkins = args.output_file_jenkins
if not valid_destination_set:
    logging.error("No valid destination(s) have been set. Exiting.")
    exit(1)


# Establish connection to mapd db
con = get_connection(
    db_user=source_db_user,
    db_passwd=source_db_passwd,
    db_server=source_db_server,
    db_port=source_db_port,
    db_name=source_db_name,
)
if not con:
    exit(1)  # Exit if cannot connect to db

# Set run vars
run_guid = str(uuid.uuid4())
logging.debug("Run guid: " + run_guid)
run_timestamp = datetime.datetime.now()
run_connection = str(con)
logging.debug("Connection string: " + run_connection)
run_driver = ""  # TODO
run_version = con._client.get_version()
if "-" in run_version:
    run_version_short = run_version.split("-")[0]
else:
    run_version_short = run_version
conn_machine_name = re.search(r"@(.*?):", run_connection).group(1)
# Set machine names, using local info if connected to localhost
if conn_machine_name == "localhost":
    local_uname = os.uname()
if machine_name:
    run_machine_name = machine_name
else:
    if conn_machine_name == "localhost":
        run_machine_name = local_uname.nodename
    else:
        run_machine_name = conn_machine_name
if machine_uname:
    run_machine_uname = machine_uname
else:
    if conn_machine_name == "localhost":
        run_machine_uname = " ".join(local_uname)
    else:
        run_machine_uname = ""

# See if import table exists, delete and/or create
if not no_drop_table_before:
    logging.info("Dropping import table if exists")
    con.execute("drop table if exists " + import_table_name)
    logging.debug("Creating import table.")
    try:
        with open(table_schema_file, "r") as table_schema:
            logging.debug("Reading table_schema_file: " + table_schema_file)
            create_table_sql = table_schema.read().replace("\n", " ")
            create_table_sql = create_table_sql.replace(
                "##TAB##", import_table_name
            )
    except FileNotFoundError:
        logging.exception("Could not find table_schema_file.")
        exit(1)
    try:
        logging.debug("Creating import table...")
        res = con.execute(create_table_sql)
        logging.debug("Import table created.")
    except (pymapd.exceptions.ProgrammingError, pymapd.exceptions.Error):
        logging.exception("Error running table creation")
        exit(1)


# Run the import query
if import_query_template_file:
    try:
        with open(import_query_template_file, "r") as import_query_template:
            logging.debug(
                "Reading import_query_template_file: " + import_query_template_file
            )
            import_query = import_query_template.read().replace("\n", " ")
            import_query = import_query.replace(
                "##TAB##", import_table_name
            )
            import_query = import_query.replace(
                "##FILE##", import_file
            )
    except FileNotFoundError:
        logging.exception("Could not find import_query_template_file.")
        exit(1)
else:
    import_query = "COPY %s FROM '%s';" % (import_table_name, import_file)
logging.debug("Import query: " + import_query)
logging.info("Starting import...")
start_time = timeit.default_timer()
try:
    res = con.execute(import_query)
    end_time = timeit.default_timer()
    logging.info("Completed import.")
except (pymapd.exceptions.ProgrammingError, pymapd.exceptions.Error):
    logging.exception("Error running import query")
    if no_drop_table_before:
        logging.error(
            'Import failed and "--no-drop-table-before" was '
            + "passed in. Make sure existing table schema matches "
            + "import .csv file schema."
        )
    exit(1)

# Calculate times and results values
query_elapsed_time = round(((end_time - start_time) * 1000), 1)
execution_time = res._result.execution_time_ms
connect_time = round((query_elapsed_time - execution_time), 2)
res_output = str(res.fetchall()[0])
logging.debug("Query result output: " + res_output)
rows_loaded = re.search(r"Loaded: (.*?) recs, R", res_output).group(1)
rows_rejected = re.search(r"Rejected: (.*?) recs i", res_output).group(1)

# Done with import. Dropping table and closing connection
if not no_drop_table_after:
    logging.debug("Dropping import table")
    con.execute("drop table " + import_table_name)
logging.debug("Closing source db connection.")
con.close()

# Update query dict entry with all values
result = {
    "run_guid": run_guid,
    "run_timestamp": run_timestamp,
    "run_connection": run_connection,
    "run_machine_name": run_machine_name,
    "run_machine_uname": run_machine_uname,
    "run_driver": run_driver,
    "run_version": run_version,
    "run_label": label,
    "import_test_name": import_test_name,
    "import_elapsed_time_ms": query_elapsed_time,
    "import_execute_time_ms": execution_time,
    "import_conn_time_ms": connect_time,
    "rows_loaded": rows_loaded,
    "rows_rejected": rows_rejected,
}


# Convert query list to json for outputs
result_json = json.dumps(result, default=json_format_handler, indent=2)

# Send results
if "mapd_db" in destinations:
    # Create dataframe from list of query results
    logging.debug("Converting results list to pandas dataframe")
    results_df = pandas.DataFrame(result, index=[0])
    # Establish connection to destination mapd db
    logging.debug("Connecting to destination mapd db")
    dest_con = get_connection(
        db_user=dest_db_user,
        db_passwd=dest_db_passwd,
        db_server=dest_db_server,
        db_port=dest_db_port,
        db_name=dest_db_name,
    )
    if not dest_con:
        exit(1)  # Exit if cannot connect to destination db
    # Load results into db, creating table if it does not exist
    tables = dest_con.get_tables()
    if dest_table not in tables:
        logging.info("Destination table does not exist. Creating.")
        try:
            with open(dest_table_schema_file, "r") as table_schema:
                logging.debug(
                    "Reading table_schema_file: " + dest_table_schema_file
                )
                create_table_sql = table_schema.read().replace("\n", " ")
                create_table_sql = create_table_sql.replace(
                    "##TAB##", dest_table
                )
        except FileNotFoundError:
            logging.exception("Could not find table_schema_file.")
            exit(1)
        try:
            logging.debug("Executing create destination table query")
            res = dest_con.execute(create_table_sql)
            logging.debug("Destination table created.")
        except (pymapd.exceptions.ProgrammingError, pymapd.exceptions.Error):
            logging.exception("Error running table creation")
            exit(1)
    logging.info("Loading results into destination db")
    dest_con.load_table(
        dest_table, results_df, method="columnar", create=False
    )
    dest_con.close()
if "file_json" in destinations:
    # Write to json file
    logging.debug("Opening json output file for writing")
    file_json_open = open(output_file_json, "w")
    logging.info("Writing to output json file: " + output_file_json)
    file_json_open.write(result_json)
if "jenkins_bench" in destinations:
    # Write output to file formatted for jenkins benchmark plugin
    # https://github.com/jenkinsci/benchmark-plugin
    logging.debug("Constructing output for jenkins benchmark plugin")
    jenkins_bench_json = json.dumps(
        {
            "groups": [
                {
                    "name": import_test_name,
                    "description": "Import: " + import_test_name,
                    "tests": [
                        {
                            "name": "import",
                            "description": "",
                            "parameters": [],
                            "results": [
                                {
                                    "name": import_test_name + " average",
                                    "description": "",
                                    "unit": "ms",
                                    "dblValue": execution_time,
                                }
                            ],
                        }
                    ],
                }
            ]
        }
    )
    # Write to json file
    logging.debug("Opening jenkins_bench json output file for writing")
    file_jenkins_open = open(output_file_jenkins, "w")
    logging.info("Writing to jenkins_bench json file: " + output_file_jenkins)
    file_jenkins_open.write(jenkins_bench_json)
if "output" in destinations:
    logging.info("Printing query results to output")
    print(result_json)
