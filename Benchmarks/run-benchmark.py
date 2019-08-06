import os
import timeit
import logging
import uuid
import datetime
import json
import numpy
import pymapd
import re
from pandas import DataFrame
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


def validate_query_file(**kwargs):
    """
      Validates query file. Currently only checks the query file name

      Kwargs:
        query_filename(str): Name of query file

      Returns:
        True(bool): Query succesfully validated
        False(bool): Query failed validation
    """
    if not kwargs["query_filename"].endswith(".sql"):
        logging.warning(
            "Query filename "
            + kwargs["query_filename"]
            + ' is invalid - does not end in ".sql". Skipping'
        )
        return False
    else:
        return True


def execute_query(**kwargs):
    """
      Executes a query against the connected db using pymapd
      https://pymapd.readthedocs.io/en/latest/usage.html#querying

      Kwargs:
        query_name(str): Name of query
        query_mapdql(str): Query to run
        iteration(int): Iteration number

      Returns:
        query_execution(dict):::
          result_count(int): Number of results returned
          execution_time(float): Time (in ms) that pymapd reports
                                 backend spent on query.
          connect_time(float): Time (in ms) for overhead of query, calculated
                               by subtracting backend execution time
                               from time spent on the execution function.
          results_iter_time(float): Time (in ms) it took to for
                                    pymapd.fetchone() to iterate through all
                                    of the results.
          total_time(float): Time (in ms) from adding all above times.
        False(bool): The query failed. Exception should be logged.
    """
    start_time = timeit.default_timer()
    try:
        # Run the query
        query_result = con.execute(kwargs["query_mapdql"])
        logging.debug(
            "Completed iteration "
            + str(kwargs["iteration"])
            + " of query "
            + kwargs["query_name"]
        )
    except (pymapd.exceptions.ProgrammingError, pymapd.exceptions.Error):
        logging.exception(
            "Error running query "
            + kwargs["query_name"]
            + " during iteration "
            + str(kwargs["iteration"])
        )
        return False

    # Calculate times
    query_elapsed_time = (timeit.default_timer() - start_time) * 1000
    execution_time = query_result._result.execution_time_ms
    connect_time = round((query_elapsed_time - execution_time), 1)

    # Iterate through each result from the query
    logging.debug(
        "Counting results from query"
        + kwargs["query_name"]
        + " iteration "
        + str(kwargs["iteration"])
    )
    result_count = 0
    start_time = timeit.default_timer()
    while query_result.fetchone():
        result_count += 1
    results_iter_time = round(
        ((timeit.default_timer() - start_time) * 1000), 1
    )

    query_execution = {
        "result_count": result_count,
        "execution_time": execution_time,
        "connect_time": connect_time,
        "results_iter_time": results_iter_time,
        "total_time": execution_time + connect_time + results_iter_time,
    }
    logging.debug(
        "Execution results for query"
        + kwargs["query_name"]
        + " iteration "
        + str(kwargs["iteration"])
        + ": "
        + str(query_execution)
    )
    return query_execution


def calculate_query_times(**kwargs):
    """
      Calculates aggregate query times from all iteration times

      Kwargs:
        total_times(list): List of total time calculations
        execution_times(list): List of execution_time calculations
        results_iter_times(list): List of results_iter_time calculations
        connect_times(list): List of connect_time calculations

      Returns:
        query_execution(dict): Query times
        False(bool): The query failed. Exception should be logged.
    """
    return {
        "total_time_avg": round(numpy.mean(kwargs["total_times"]), 1),
        "total_time_min": round(numpy.min(kwargs["total_times"]), 1),
        "total_time_max": round(numpy.max(kwargs["total_times"]), 1),
        "total_time_85": round(numpy.percentile(kwargs["total_times"], 85), 1),
        "execution_time_avg": round(numpy.mean(kwargs["execution_times"]), 1),
        "execution_time_min": round(numpy.min(kwargs["execution_times"]), 1),
        "execution_time_max": round(numpy.max(kwargs["execution_times"]), 1),
        "execution_time_85": round(
            numpy.percentile(kwargs["execution_times"], 85), 1
        ),
        "execution_time_25": round(
            numpy.percentile(kwargs["execution_times"], 25), 1
        ),
        "execution_time_std": round(numpy.std(kwargs["execution_times"]), 1),
        "connect_time_avg": round(numpy.mean(kwargs["connect_times"]), 1),
        "connect_time_min": round(numpy.min(kwargs["connect_times"]), 1),
        "connect_time_max": round(numpy.max(kwargs["connect_times"]), 1),
        "connect_time_85": round(
            numpy.percentile(kwargs["connect_times"], 85), 1
        ),
        "results_iter_time_avg": round(
            numpy.mean(kwargs["results_iter_times"]), 1
        ),
        "results_iter_time_min": round(
            numpy.min(kwargs["results_iter_times"]), 1
        ),
        "results_iter_time_max": round(
            numpy.max(kwargs["results_iter_times"]), 1
        ),
        "results_iter_time_85": round(
            numpy.percentile(kwargs["results_iter_times"], 85), 1
        ),
    }


def get_mem_usage(**kwargs):
    """
      Calculates memory statistics from mapd_server _client.get_memory call

      Kwargs:
        con(class 'pymapd.connection.Connection'): Mapd connection
        mem_type(str): [gpu, cpu] Type of memory to gather metrics for

      Returns:
        ramusage(dict):::
          usedram(float): Amount of memory (in MB) used
          freeram(float): Amount of memory (in MB) free
          totalallocated(float): Total amount of memory (in MB) allocated
          errormessage(str): Error if returned by get_memory call
          rawdata(list): Raw data returned from get_memory call
    """
    try:
        con_mem_data_list = con._client.get_memory(
            session=kwargs["con"]._session, memory_level=kwargs["mem_type"]
        )
        usedram = 0
        freeram = 0
        for con_mem_data in con_mem_data_list:
            page_size = con_mem_data.page_size
            node_memory_data_list = con_mem_data.node_memory_data
            for node_memory_data in node_memory_data_list:
                ram = node_memory_data.num_pages * page_size
                is_free = node_memory_data.is_free
                if is_free:
                    freeram += ram
                else:
                    usedram += ram
        totalallocated = usedram + freeram
        if totalallocated > 0:
            totalallocated = round(totalallocated / 1024 / 1024, 1)
            usedram = round(usedram / 1024 / 1024, 1)
            freeram = round(freeram / 1024 / 1024, 1)
        ramusage = {}
        ramusage["usedram"] = usedram
        ramusage["freeram"] = freeram
        ramusage["totalallocated"] = totalallocated
        ramusage["errormessage"] = ""
    except Exception as e:
        errormessage = "Get memory failed with error: " + str(e)
        logging.error(errormessage)
        ramusage["errormessage"] = errormessage
    return ramusage


def json_format_handler(x):
    # Function to allow json to deal with datetime and numpy int
    if isinstance(x, datetime.datetime):
        return x.isoformat()
    if isinstance(x, numpy.int64):
        return int(x)
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
    "-t", "--table", dest="table", required=True, help="Source db table name"
)
required.add_argument(
    "-l", "--label", dest="label", required=True, help="Benchmark run label"
)
required.add_argument(
    "-d",
    "--queries-dir",
    dest="queries_dir",
    help='Absolute path to dir with query files. \
                      [Default: "queries" dir in same location as script]',
)
required.add_argument(
    "-i",
    "--iterations",
    dest="iterations",
    type=int,
    required=True,
    help="Number of iterations per query. Must be > 1",
)
optional.add_argument(
    "-g",
    "--gpu-count",
    dest="gpu_count",
    type=int,
    default=None,
    help="Number of GPUs. Not required when gathering local gpu info",
)
optional.add_argument(
    "-G",
    "--gpu-name",
    dest="gpu_name",
    type=str,
    default="",
    help="Name of GPU(s). Not required when gathering local gpu info",
)
optional.add_argument(
    "--no-gather-conn-gpu-info",
    dest="no_gather_conn_gpu_info",
    action="store_true",
    help="Do not gather source database GPU info fields "
    + "[run_gpu_count, run_gpu_mem_mb] "
    + "using pymapd connection info. "
    + "Use when testing a CPU-only server.",
)
optional.add_argument(
    "--no-gather-nvml-gpu-info",
    dest="no_gather_nvml_gpu_info",
    action="store_true",
    help="Do not gather source database GPU info fields "
    + "[gpu_driver_ver, run_gpu_name] "
    + "from local GPU using pynvml. "
    + 'Defaults to True when source server is not "localhost". '
    + "Use when testing a CPU-only server.",
)
optional.add_argument(
    "--gather-nvml-gpu-info",
    dest="gather_nvml_gpu_info",
    action="store_true",
    help="Gather source database GPU info fields "
    + "[gpu_driver_ver, run_gpu_name] "
    + "from local GPU using pynvml. "
    + 'Defaults to True when source server is "localhost". '
    + "Only use when benchmarking against same machine that this script is "
    + "run from.",
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
    default="results",
    help="Destination mapd_db table name",
)
optional.add_argument(
    "-C",
    "--dest-table-schema-file",
    dest="dest_table_schema_file",
    default="results_table_schemas/query-results.sql",
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
optional.add_argument(
    "-E",
    "--output-tag-jenkins",
    dest="output_tag_jenkins",
    default="",
    help="Jenkins benchmark result tag. "
    + 'Optional, appended to table name in "group" field',
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
source_table = args.table
label = args.label
if args.queries_dir:
    queries_dir = args.queries_dir
else:
    queries_dir = os.path.join(os.path.dirname(__file__), "queries")
iterations = int(args.iterations)
if (iterations > 1) is not True:
    # Need > 1 iteration as first iteration is dropped from calculations
    logging.error("Iterations must be greater than 1")
    exit(1)
gpu_count = args.gpu_count
gpu_name = args.gpu_name
no_gather_conn_gpu_info = args.no_gather_conn_gpu_info
gather_nvml_gpu_info = args.gather_nvml_gpu_info
no_gather_nvml_gpu_info = args.no_gather_nvml_gpu_info
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
    else:
        output_file_jenkins = args.output_file_jenkins
output_tag_jenkins = args.output_tag_jenkins
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
# Set GPU info fields
conn_gpu_count = None
source_db_gpu_count = None
source_db_gpu_mem = None
source_db_gpu_driver_ver = ""
source_db_gpu_name = ""
if no_gather_conn_gpu_info:
    logging.debug(
        "--no-gather-conn-gpu-info passed, "
        + "using blank values for source database GPU info fields "
        + "[run_gpu_count, run_gpu_mem_mb] "
    )
else:
    logging.debug(
        "Gathering source database GPU info fields "
        + "[run_gpu_count, run_gpu_mem_mb] "
        + "using pymapd connection info. "
    )
    conn_hardware_info = con._client.get_hardware_info(con._session)
    conn_gpu_count = conn_hardware_info.hardware_info[0].num_gpu_allocated
if conn_gpu_count == 0 or conn_gpu_count is None:
    no_gather_nvml_gpu_info = True
    if conn_gpu_count == 0:
        logging.warning(
            "0 GPUs detected from connection info, "
            + "using blank values for source database GPU info fields "
            + "If running against cpu-only server, make sure to set "
            + "--no-gather-nvml-gpu-info and --no-gather-conn-gpu-info."
        )
else:
    source_db_gpu_count = conn_gpu_count
    try:
        source_db_gpu_mem = int(
            conn_hardware_info.hardware_info[0].gpu_info[0].memory / 1000000
        )
    except IndexError:
        logging.error("GPU memory info not available from connection.")
if no_gather_nvml_gpu_info:
    logging.debug(
        "--no-gather-nvml-gpu-info passed, "
        + "using blank values for source database GPU info fields "
        + "[gpu_driver_ver, run_gpu_name] "
    )
elif conn_machine_name == "localhost" or gather_nvml_gpu_info:
    logging.debug(
        "Gathering source database GPU info fields "
        + "[gpu_driver_ver, run_gpu_name] "
        + "from local GPU using pynvml. "
    )
    import pynvml

    pynvml.nvmlInit()
    source_db_gpu_driver_ver = pynvml.nvmlSystemGetDriverVersion().decode()
    for i in range(source_db_gpu_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        # Assume all cards are the same, overwrite name value
        source_db_gpu_name = pynvml.nvmlDeviceGetName(handle).decode()
    pynvml.nvmlShutdown()
# If gpu_count argument passed in, override gathered value
if gpu_count:
    source_db_gpu_count = gpu_count
# Set machine names, using local info if connected to localhost
if conn_machine_name == "localhost":
    local_uname = os.uname()
if machine_name:
    run_machine_name = machine_name
else:
    if conn_machine_name == "localhost":
        run_machine_name = local_uname.nodename.split(".")[0]
    else:
        run_machine_name = conn_machine_name
if machine_uname:
    run_machine_uname = machine_uname
else:
    if conn_machine_name == "localhost":
        run_machine_uname = " ".join(local_uname)
    else:
        run_machine_uname = ""

# Read query files contents and write to query_list
query_list = []
logging.debug("Queries dir: " + queries_dir)
try:
    for query_filename in os.listdir(queries_dir):
        logging.debug("Validating query filename: " + query_filename)
        if validate_query_file(query_filename=query_filename):
            with open(
                queries_dir + "/" + query_filename, "r"
            ) as query_filepath:
                logging.debug("Reading query with filename: " + query_filename)
                query_mapdql = query_filepath.read().replace("\n", " ")
                query_mapdql = query_mapdql.replace("##TAB##", source_table)
                query_list.append(
                    {"name": query_filename, "mapdql": query_mapdql}
                )
    logging.info("Read all query files")
except FileNotFoundError:
    logging.exception("Could not find queries directory.")
    exit(1)  # Exit if cannot get queries dir

# Run queries
for query in query_list:
    # Set additional query vars
    # Query ID = filename without extention
    query_id = query["name"].rsplit(".")[0]

    # Run iterations of query
    query_results = []
    logging.info(
        "Running query: " + query["name"] + " iterations: " + str(iterations)
    )
    query_total_start_time = timeit.default_timer()
    for iteration in range(iterations):
        # Gather memory before running query iteration
        logging.debug("Getting pre-query memory usage on CPU")
        pre_query_cpu_mem_usage = get_mem_usage(con=con, mem_type="cpu")
        logging.debug("Getting pre-query memory usage on GPU")
        pre_query_gpu_mem_usage = get_mem_usage(con=con, mem_type="gpu")
        # Run query iteration
        logging.debug(
            "Running iteration "
            + str(iteration)
            + " of query "
            + query["name"]
        )
        query_result = execute_query(
            query_name=query["name"],
            query_mapdql=query["mapdql"],
            iteration=iteration,
        )
        # Gather memory after running query iteration
        logging.debug("Getting post-query memory usage on CPU")
        post_query_cpu_mem_usage = get_mem_usage(con=con, mem_type="cpu")
        logging.debug("Getting post-query memory usage on GPU")
        post_query_gpu_mem_usage = get_mem_usage(con=con, mem_type="gpu")
        # Calculate total (post minus pre) memory usage after query iteration
        query_cpu_mem_usage = round(
            post_query_cpu_mem_usage["usedram"]
            - pre_query_cpu_mem_usage["usedram"],
            1,
        )
        query_gpu_mem_usage = round(
            post_query_gpu_mem_usage["usedram"]
            - pre_query_gpu_mem_usage["usedram"],
            1,
        )
        if query_result:
            query.update({"succeeded": True})
            query_error_info = ""  # TODO - interpret query error info
            # Assign first query iteration times
            if iteration == 0:
                first_execution_time = round(query_result["execution_time"], 1)
                first_connect_time = round(query_result["connect_time"], 1)
                first_results_iter_time = round(
                    query_result["results_iter_time"], 1
                )
                first_total_time = (
                    first_execution_time
                    + first_connect_time
                    + first_results_iter_time
                )
                first_cpu_mem_usage = query_cpu_mem_usage
                first_gpu_mem_usage = query_gpu_mem_usage
            else:
                # Put noninitial iterations into query_result list
                query_results.append(query_result)
                # Verify no change in memory for noninitial iterations
                if query_cpu_mem_usage != 0.0:
                    logging.error(
                        (
                            "Noninitial iteration ({0}) of query ({1}) "
                            + "shows non-zero CPU memory usage: {2}"
                        ).format(iteration, query["name"], query_cpu_mem_usage)
                    )
                if query_gpu_mem_usage != 0.0:
                    logging.error(
                        (
                            "Noninitial iteration ({0}) of query ({1}) "
                            + "shows non-zero GPU memory usage: {2}"
                        ).format(iteration, query["name"], query_gpu_mem_usage)
                    )
        else:
            query.update({"succeeded": False})
            logging.warning(
                "Error detected during execution of query: "
                + query["name"]
                + ". This query will be skipped and "
                + "times will not reported"
            )
        if query["succeeded"] is False:
            # Do not run any more iterations of the failed query
            break
    if query["succeeded"] is False:
        # Do not calculate results for the failed query, move on to the next
        continue

    # Calculate time for all iterations to run
    query_total_elapsed_time = round(
        ((timeit.default_timer() - query_total_start_time) * 1000), 1
    )
    logging.info("Completed all iterations of query " + query["name"])

    # Aggregate iteration values
    execution_times, connect_times, results_iter_times, total_times = (
        [],
        [],
        [],
        [],
    )
    for query_result in query_results:
        execution_times.append(query_result["execution_time"])
        connect_times.append(query_result["connect_time"])
        results_iter_times.append(query_result["results_iter_time"])
        total_times.append(query_result["total_time"])
        # Overwrite result count, since should be the same for each iteration
        result_count = query_result["result_count"]

    # Calculate query times
    logging.debug("Calculating times from query " + query["name"])
    query_times = calculate_query_times(
        total_times=total_times,
        execution_times=execution_times,
        connect_times=connect_times,
        results_iter_times=results_iter_times,
    )

    # Update query dict entry with all values
    query.update(
        {
            "results": {
                "run_guid": run_guid,
                "run_timestamp": run_timestamp,
                "run_connection": run_connection,
                "run_machine_name": run_machine_name,
                "run_machine_uname": run_machine_uname,
                "run_driver": run_driver,
                "run_version": run_version,
                "run_version_short": run_version_short,
                "run_label": label,
                "run_gpu_count": source_db_gpu_count,
                "run_gpu_driver_ver": source_db_gpu_driver_ver,
                "run_gpu_name": source_db_gpu_name,
                "run_gpu_mem_mb": source_db_gpu_mem,
                "run_table": source_table,
                "query_id": query_id,
                "query_result_set_count": result_count,
                "query_error_info": query_error_info,
                "query_conn_first": first_connect_time,
                "query_conn_avg": query_times["connect_time_avg"],
                "query_conn_min": query_times["connect_time_min"],
                "query_conn_max": query_times["connect_time_max"],
                "query_conn_85": query_times["connect_time_85"],
                "query_exec_first": first_execution_time,
                "query_exec_avg": query_times["execution_time_avg"],
                "query_exec_min": query_times["execution_time_min"],
                "query_exec_max": query_times["execution_time_max"],
                "query_exec_85": query_times["execution_time_85"],
                "query_exec_25": query_times["execution_time_25"],
                "query_exec_stdd": query_times["execution_time_std"],
                # Render queries not supported yet
                "query_render_first": None,
                "query_render_avg": None,
                "query_render_min": None,
                "query_render_max": None,
                "query_render_85": None,
                "query_render_25": None,
                "query_render_stdd": None,
                "query_total_first": first_total_time,
                "query_total_avg": query_times["total_time_avg"],
                "query_total_min": query_times["total_time_min"],
                "query_total_max": query_times["total_time_max"],
                "query_total_85": query_times["total_time_85"],
                "query_total_all": query_total_elapsed_time,
                "results_iter_count": iterations,
                "results_iter_first": first_results_iter_time,
                "results_iter_avg": query_times["results_iter_time_avg"],
                "results_iter_min": query_times["results_iter_time_min"],
                "results_iter_max": query_times["results_iter_time_max"],
                "results_iter_85": query_times["results_iter_time_85"],
                "cpu_mem_usage_mb": first_cpu_mem_usage,
                "gpu_mem_usage_mb": first_gpu_mem_usage,
            }
        }
    )
    logging.debug(
        "All values set for query " + query["name"] + ": " + str(query)
    )
logging.debug("Closing source db connection.")
con.close()
logging.info("Completed all queries.")


# Create list of successful queries
logging.debug(
    "Removing failed queries from results going to destination db(s)"
)
succesful_query_list = query_list
for index, query in enumerate(succesful_query_list):
    if query["succeeded"] is False:
        del succesful_query_list[index]
# Create successful query results list for upload to destination(s)
query_results = []
for query in succesful_query_list:
    query_results.append(query["results"])
# Convert query list to json for outputs
query_list_json = json.dumps(query_list, default=json_format_handler, indent=2)

# Send results
if "mapd_db" in destinations:
    # Create dataframe from list of query results
    logging.debug("Converting results list to pandas dataframe")
    results_df = DataFrame(query_results)
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
    dest_con.load_table_columnar(
        dest_table,
        results_df,
        preserve_index=False,
        chunk_size_bytes=0,
        col_names_from_schema=True,
    )
    dest_con.close()
if "file_json" in destinations:
    # Write to json file
    logging.debug("Opening json output file for writing")
    file_json_open = open(output_file_json, "w")
    logging.info("Writing to output json file: " + output_file_json)
    file_json_open.write(query_list_json)
if "jenkins_bench" in destinations:
    # Write output to file formatted for jenkins benchmark plugin
    # https://github.com/jenkinsci/benchmark-plugin
    jenkins_bench_results = []
    for query_result in query_results:
        logging.debug("Constructing output for jenkins benchmark plugin")
        jenkins_bench_results.append(
            {
                "name": query_result["query_id"],
                "description": "",
                "parameters": [],
                "results": [
                    {
                        "name": query_result["query_id"] + " average",
                        "description": "",
                        "unit": "ms",
                        "dblValue": query_result["query_exec_avg"],
                    }
                ],
            }
        )
    jenkins_bench_json = json.dumps(
        {
            "groups": [
                {
                    "name": source_table + output_tag_jenkins,
                    "description": "Source table: " + source_table,
                    "tests": jenkins_bench_results,
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
    print(query_list_json)

logging.info("Succesfully loaded query results info into destination(s)")
