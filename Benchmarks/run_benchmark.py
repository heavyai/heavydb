import os
import timeit
import logging
import uuid
import datetime
import json
import numpy
import pymapd
import re
import sys
from pandas import DataFrame
from argparse import ArgumentParser

# For usage info, run: `./<script_name>.py --help`


def verify_destinations(**kwargs):
    """
      Verify script output destination(s)

      Kwargs:
        destinations (list): List of destinations
        dest_db_server (str): DB output destination server
        output_file_json (str): Location of .json file output
        output_file_jenkins (str): Location of .json jenkins file output

      Returns:
        True(bool): Destination(s) is/are valid
        False(bool): Destination(s) is/are not valid
    """
    if "mapd_db" in kwargs["destinations"]:
        valid_destination_set = True
        if kwargs["dest_db_server"] is None:
            # If dest_server is not set for mapd_db, then exit
            logging.error(
                '"dest_server" is required when destination = "mapd_db"'
            )
    if "file_json" in kwargs["destinations"]:
        valid_destination_set = True
        if kwargs["output_file_json"] is None:
            # If output_file_json is not set for file_json, then exit
            logging.error(
                '"output_file_json" is required when destination = "file_json"'
            )
    if "output" in kwargs["destinations"]:
        valid_destination_set = True
    if "jenkins_bench" in kwargs["destinations"]:
        valid_destination_set = True
        if kwargs["output_file_jenkins"] is None:
            # If output_file_jenkins is not set for jenkins_bench, then exit
            logging.error(
                '"output_file_jenkins" is required '
                + 'when destination = "jenkins_bench"'
            )
    if not valid_destination_set:
        return False
    else:
        return True


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


def get_run_vars(**kwargs):
    """
      Gets/sets run-specific vars such as time, uid, etc.

      Kwargs:
        con(class 'pymapd.connection.Connection'): Mapd connection

      Returns:
        run_vars(dict):::
            run_guid(str): Run GUID
            run_timestamp(datetime): Run timestamp
            run_connection(str): Connection string
            run_driver(str): Run driver
            run_version(str): Version of DB
            run_version_short(str): Shortened version of DB
            conn_machine_name(str): Name of run machine
    """
    run_guid = str(uuid.uuid4())
    logging.debug("Run guid: " + run_guid)
    run_timestamp = datetime.datetime.now()
    run_connection = str(kwargs["con"])
    logging.debug("Connection string: " + run_connection)
    run_driver = ""  # TODO
    run_version = kwargs["con"]._client.get_version()
    if "-" in run_version:
        run_version_short = run_version.split("-")[0]
    else:
        run_version_short = run_version
    conn_machine_name = re.search(r"@(.*?):", run_connection).group(1)
    run_vars = {
        "run_guid": run_guid,
        "run_timestamp": run_timestamp,
        "run_connection": run_connection,
        "run_driver": run_driver,
        "run_version": run_version,
        "run_version_short": run_version_short,
        "conn_machine_name": conn_machine_name,
    }
    return run_vars


def get_gpu_info(**kwargs):
    """
      Gets run machine GPU info

      Kwargs:
        gpu_name(str): GPU name from input param
        no_gather_conn_gpu_info(bool): Gather GPU info fields
        con(class 'pymapd.connection.Connection'): Mapd connection
        conn_machine_name(str): Name of run machine
        no_gather_nvml_gpu_info(bool): Do not gather GPU info using nvml
        gather_nvml_gpu_info(bool): Gather GPU info using nvml
        gpu_count(int): Number of GPUs on run machine

      Returns:
        gpu_info(dict):::
            conn_gpu_count(int): Number of GPUs gathered from pymapd con
            source_db_gpu_count(int): Number of GPUs on run machine
            source_db_gpu_mem(str): Amount of GPU mem on run machine
            source_db_gpu_driver_ver(str): GPU driver version
            source_db_gpu_name(str): GPU name
    """
    # Set GPU info fields
    conn_gpu_count = None
    source_db_gpu_count = None
    source_db_gpu_mem = None
    source_db_gpu_driver_ver = ""
    source_db_gpu_name = ""
    if kwargs["no_gather_conn_gpu_info"]:
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
        conn_hardware_info = kwargs["con"]._client.get_hardware_info(
            kwargs["con"]._session
        )
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
        no_gather_nvml_gpu_info = kwargs["no_gather_nvml_gpu_info"]
        source_db_gpu_count = conn_gpu_count
        try:
            source_db_gpu_mem = int(
                conn_hardware_info.hardware_info[0].gpu_info[0].memory
                / 1000000
            )
        except IndexError:
            logging.error("GPU memory info not available from connection.")
    if no_gather_nvml_gpu_info:
        logging.debug(
            "--no-gather-nvml-gpu-info passed, "
            + "using blank values for source database GPU info fields "
            + "[gpu_driver_ver, run_gpu_name] "
        )
    elif (
        kwargs["conn_machine_name"] == "localhost"
        or kwargs["gather_nvml_gpu_info"]
    ):
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
    if kwargs["gpu_count"]:
        source_db_gpu_count = kwargs["gpu_count"]
    if kwargs["gpu_name"]:
        source_db_gpu_name = kwargs["gpu_name"]
    gpu_info = {
        "conn_gpu_count": conn_gpu_count,
        "source_db_gpu_count": source_db_gpu_count,
        "source_db_gpu_mem": source_db_gpu_mem,
        "source_db_gpu_driver_ver": source_db_gpu_driver_ver,
        "source_db_gpu_name": source_db_gpu_name,
    }
    return gpu_info


def get_machine_info(**kwargs):
    """
      Gets run machine GPU info

      Kwargs:
        conn_machine_name(str): Name of machine from pymapd con
        machine_name(str): Name of machine if passed in
        machine_uname(str): Uname of machine if passed in

      Returns:
        machine_info(dict):::
            run_machine_name(str): Run machine name
            run_machine_uname(str): Run machine uname
    """
    # Set machine names, using local info if connected to localhost
    if kwargs["conn_machine_name"] == "localhost":
        local_uname = os.uname()
    # If --machine-name passed in, override pymapd con value
    if kwargs["machine_name"]:
        run_machine_name = kwargs["machine_name"]
    else:
        if kwargs["conn_machine_name"] == "localhost":
            run_machine_name = local_uname.nodename.split(".")[0]
        else:
            run_machine_name = kwargs["conn_machine_name"]
    # If --machine-uname passed in, override pymapd con value
    if kwargs["machine_uname"]:
        run_machine_uname = kwargs["machine_uname"]
    else:
        if kwargs["conn_machine_name"] == "localhost":
            run_machine_uname = " ".join(local_uname)
        else:
            run_machine_uname = ""
    machine_info = {
        "run_machine_name": run_machine_name,
        "run_machine_uname": run_machine_uname,
    }
    return machine_info


def read_query_files(**kwargs):
    """
      Gets run machine GPU info

      Kwargs:
        queries_dir(str): Directory with query files
        source_table(str): Table to run query against

      Returns:
        query_list(dict):::
            query_group(str): Query group, usually matches table name
            queries(list)
                query(dict):::
                    name(str): Name of query
                    mapdql(str): Query syntax to run
        False(bool): Unable to find queries dir
    """
    # Read query files contents and write to query_list
    query_list = {"query_group": "", "queries": []}
    query_group = kwargs["queries_dir"].split("/")[-1]
    query_list.update(query_group=query_group)
    logging.debug("Queries dir: " + kwargs["queries_dir"])
    try:
        for query_filename in sorted(os.listdir(kwargs["queries_dir"])):
            logging.debug("Validating query filename: " + query_filename)
            if validate_query_file(query_filename=query_filename):
                with open(
                    kwargs["queries_dir"] + "/" + query_filename, "r"
                ) as query_filepath:
                    logging.debug(
                        "Reading query with filename: " + query_filename
                    )
                    query_mapdql = query_filepath.read().replace("\n", " ")
                    query_mapdql = query_mapdql.replace(
                        "##TAB##", kwargs["source_table"]
                    )
                    query_list["queries"].append(
                        {"name": query_filename, "mapdql": query_mapdql}
                    )
        logging.info("Read all query files")
        return query_list
    except FileNotFoundError:
        logging.exception("Could not find queries directory.")
        return False


def read_setup_teardown_query_files(**kwargs):
    """
      Get queries to run for setup and teardown from directory

      Kwargs:
        queries_dir(str): Directory with query files
        source_table(str): Table to run query against
        foreign_table_filename(str): File to create foreign table from

      Returns:
        setup_queries(query_list): List of setup queries
        teardown_queries(query_list): List of teardown queries
        False(bool): Unable to find queries dir

    query_list is described by:
    query_list(dict):::
        query_group(str): Query group, usually matches table name
        queries(list)
            query(dict):::
                name(str): Name of query
                mapdql(str): Query syntax to run
    """
    setup_teardown_queries_dir = kwargs['queries_dir']
    source_table = kwargs['source_table']
    # Read setup/tear-down queries if they exist
    setup_teardown_query_list = None
    if setup_teardown_queries_dir is not None:
        setup_teardown_query_list = read_query_files(
            queries_dir=setup_teardown_queries_dir,
            source_table=source_table
        )
        if kwargs["foreign_table_filename"] is not None:
            for query in setup_teardown_query_list['queries']:
                query['mapdql'] = query['mapdql'].replace(
                    "##FILE##", kwargs["foreign_table_filename"])
    # Filter setup queries
    setup_query_list = None
    if setup_teardown_query_list is not None:
        setup_query_list = filter(
            lambda x: validate_setup_teardown_query_file(
                query_filename=x['name'], check_which='setup', quiet=True),
            setup_teardown_query_list['queries'])
        setup_query_list = list(setup_query_list)
    # Filter teardown queries
    teardown_query_list = None
    if setup_teardown_query_list is not None:
        teardown_query_list = filter(
            lambda x: validate_setup_teardown_query_file(
                query_filename=x['name'], check_which='teardown', quiet=True),
            setup_teardown_query_list['queries'])
        teardown_query_list = list(teardown_query_list)
    return setup_query_list, teardown_query_list


def validate_setup_teardown_query_file(**kwargs):
    """
      Validates query file. Currently only checks the query file name, and
      checks for setup or teardown in basename

      Kwargs:
        query_filename(str): Name of query file
        check_which(bool): either 'setup' or 'teardown', decide which to
                           check
        quiet(bool): optional, if True, no warning is logged

      Returns:
        True(bool): Query succesfully validated
        False(bool): Query failed validation
    """
    qfilename = kwargs["query_filename"]
    basename = os.path.basename(qfilename)
    check_str = False
    if kwargs["check_which"] == 'setup':
        check_str = basename.lower().find('setup') > -1
    elif kwargs["check_which"] == 'teardown':
        check_str = basename.lower().find('teardown') > -1
    else:
        raise TypeError('Unsupported `check_which` parameter.')
    return_val = True
    if not qfilename.endswith(".sql"):
        logging.warning(
            "Query filename "
            + qfilename
            + ' is invalid - does not end in ".sql". Skipping'
        )
        return_val = False
    elif not check_str:
        quiet = True if 'quiet' in kwargs and kwargs['quiet'] else False
        if not quiet:
            logging.warning(
                "Query filename "
                + qfilename
                + ' does not match "setup" or "teardown". Skipping'
            )
        return_val = False
    return return_val


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
        con(class): Connection class

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
        query_result = kwargs["con"].execute(kwargs["query_mapdql"])
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
    debug_info = query_result._result.debug
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
        "debug_info": debug_info,
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
        trim(float): Amount to trim from iterations set to gather trimmed
                     values. Enter as deciman corresponding to percent to
                     trim - ex: 0.15 to trim 15%.

      Returns:
        query_execution(dict): Query times
        False(bool): The query failed. Exception should be logged.
    """
    trim_size = int(kwargs["trim"] * len(kwargs["total_times"]))
    return {
        "total_time_avg": round(numpy.mean(kwargs["total_times"]), 1),
        "total_time_min": round(numpy.min(kwargs["total_times"]), 1),
        "total_time_max": round(numpy.max(kwargs["total_times"]), 1),
        "total_time_85": round(numpy.percentile(kwargs["total_times"], 85), 1),
        "total_time_trimmed_avg": round(
            numpy.mean(
                numpy.sort(kwargs["total_times"])[trim_size:-trim_size]
            ),
            1,
        )
        if trim_size
        else round(numpy.mean(kwargs["total_times"]), 1),
        "total_times": kwargs["total_times"],
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
        "execution_time_trimmed_avg": round(
            numpy.mean(
                numpy.sort(kwargs["execution_times"])[trim_size:-trim_size]
            )
        )
        if trim_size > 0
        else round(numpy.mean(kwargs["execution_times"]), 1),
        "execution_time_trimmed_max": round(
            numpy.max(
                numpy.sort(kwargs["execution_times"])[trim_size:-trim_size]
            )
        )
        if trim_size > 0
        else round(numpy.max(kwargs["execution_times"]), 1),
        "execution_times": kwargs["execution_times"],
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
        con_mem_data_list = kwargs["con"]._client.get_memory(
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


def run_query(**kwargs):
    """
      Takes query name, syntax, and iteration count and calls the
        execute_query function for each iteration. Reports total, iteration,
        and exec timings, memory usage, and failure status.

      Kwargs:
        query(dict):::
            name(str): Name of query
            mapdql(str): Query syntax to run
        iterations(int): Number of iterations of each query to run
        trim(float): Trim decimal to remove from top and bottom of results
        con(class 'pymapd.connection.Connection'): Mapd connection

      Returns:
        query_results(dict):::
            query_name(str): Name of query
            query_mapdql(str): Query to run
            query_id(str): Query ID
            query_succeeded(bool): Query succeeded
            query_error_info(str): Query error info
            result_count(int): Number of results returned
            initial_iteration_results(dict):::
                first_execution_time(float): Execution time for first query
                    iteration
                first_connect_time(float):  Connect time for first query
                    iteration
                first_results_iter_time(float): Results iteration time for
                    first query iteration
                first_total_time(float): Total time for first iteration
                first_cpu_mem_usage(float): CPU memory usage for first query
                    iteration
                first_gpu_mem_usage(float): GPU memory usage for first query
                    iteration
            noninitial_iteration_results(list):::
                execution_time(float): Time (in ms) that pymapd reports
                    backend spent on query.
                connect_time(float): Time (in ms) for overhead of query,
                    calculated by subtracting backend execution time from
                    time spent on the execution function.
                results_iter_time(float): Time (in ms) it took to for
                    pymapd.fetchone() to iterate through all of the results.
                total_time(float): Time (in ms) from adding all above times.
            query_total_elapsed_time(int): Total elapsed time for query
        False(bool): The query failed. Exception should be logged.
    """
    logging.info(
        "Running query: "
        + kwargs["query"]["name"]
        + " iterations: "
        + str(kwargs["iterations"])
    )
    query_id = kwargs["query"]["name"].rsplit(".")[
        0
    ]  # Query ID = filename without extention
    query_results = {
        "query_name": kwargs["query"]["name"],
        "query_mapdql": kwargs["query"]["mapdql"],
        "query_id": query_id,
        "query_succeeded": True,
        "query_error_info": "",
        "initial_iteration_results": {},
        "noninitial_iteration_results": [],
        "query_total_elapsed_time": 0,
    }
    query_total_start_time = timeit.default_timer()
    # Run iterations of query
    for iteration in range(kwargs["iterations"]):
        # Gather memory before running query iteration
        logging.debug("Getting pre-query memory usage on CPU")
        pre_query_cpu_mem_usage = get_mem_usage(
            con=kwargs["con"], mem_type="cpu"
        )
        logging.debug("Getting pre-query memory usage on GPU")
        pre_query_gpu_mem_usage = get_mem_usage(
            con=kwargs["con"], mem_type="gpu"
        )
        # Run query iteration
        logging.debug(
            "Running iteration "
            + str(iteration)
            + " of query "
            + kwargs["query"]["name"]
        )
        query_result = execute_query(
            query_name=kwargs["query"]["name"],
            query_mapdql=kwargs["query"]["mapdql"],
            iteration=iteration,
            con=kwargs["con"],
        )
        # Gather memory after running query iteration
        logging.debug("Getting post-query memory usage on CPU")
        post_query_cpu_mem_usage = get_mem_usage(
            con=kwargs["con"], mem_type="cpu"
        )
        logging.debug("Getting post-query memory usage on GPU")
        post_query_gpu_mem_usage = get_mem_usage(
            con=kwargs["con"], mem_type="gpu"
        )
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
            query_results.update(
                query_error_info=""  # TODO - interpret query error info
            )
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
                query_results.update(
                    initial_iteration_results={
                        "first_execution_time": first_execution_time,
                        "first_connect_time": first_connect_time,
                        "first_results_iter_time": first_results_iter_time,
                        "first_total_time": first_total_time,
                        "first_cpu_mem_usage": query_cpu_mem_usage,
                        "first_gpu_mem_usage": query_gpu_mem_usage,
                    }
                )
            else:
                # Put noninitial iterations into query_result list
                query_results["noninitial_iteration_results"].append(
                    query_result
                )
                # Verify no change in memory for noninitial iterations
                if query_cpu_mem_usage != 0.0:
                    logging.error(
                        (
                            "Noninitial iteration ({0}) of query ({1}) "
                            + "shows non-zero CPU memory usage: {2}"
                        ).format(
                            iteration,
                            kwargs["query"]["name"],
                            query_cpu_mem_usage,
                        )
                    )
                if query_gpu_mem_usage != 0.0:
                    logging.error(
                        (
                            "Noninitial iteration ({0}) of query ({1}) "
                            + "shows non-zero GPU memory usage: {2}"
                        ).format(
                            iteration,
                            kwargs["query"]["name"],
                            query_gpu_mem_usage,
                        )
                    )
        else:
            logging.warning(
                "Error detected during execution of query: "
                + kwargs["query"]["name"]
                + ". This query will be skipped and "
                + "times will not reported"
            )
            query_results.update(query_succeeded=False)
            break
    # Calculate time for all iterations to run
    query_total_elapsed_time = round(
        ((timeit.default_timer() - query_total_start_time) * 1000), 1
    )
    query_results.update(query_total_elapsed_time=query_total_elapsed_time)
    logging.info(
        "Completed all iterations of query " + kwargs["query"]["name"]
    )
    return query_results


def run_setup_teardown_query(**kwargs):
    """
        Convenience wrapper around `run_query` to run a setup or 
        teardown query

      Kwargs:
        queries(query_list): List of queries to run
        do_run(bool): If true will run query, otherwise do nothing
        trim(float): Trim decimal to remove from top and bottom of results
        con(class 'pymapd.connection.Connection'): Mapd connection

      Returns:
        See return value for `run_query`

        query_list is described by:
        queries(list)
            query(dict):::
                name(str): Name of query
                mapdql(str): Query syntax to run
                [setup : queries(list)]
                [teardown : queries(list)]
    """
    query_results = list()
    if kwargs['do_run']:
        for query in kwargs['queries']:
            result = run_query(
                query=query, iterations=1,
                trim=kwargs['trim'],
                con=kwargs['con']
            )
            if not result['query_succeeded']:
                logging.warning(
                    "Error setup or teardown query: "
                    + query["name"]
                    + ". did not complete."
                )
            else:
                query_results.append(result)
    return query_results


def json_format_handler(x):
    # Function to allow json to deal with datetime and numpy int
    if isinstance(x, datetime.datetime):
        return x.isoformat()
    if isinstance(x, numpy.int64):
        return int(x)
    raise TypeError("Unknown type")


def create_results_dataset(**kwargs):
    """
      Create results dataset

      Kwargs:
        run_guid(str): Run GUID
        run_timestamp(datetime): Run timestamp
        run_connection(str): Connection string
        run_machine_name(str): Run machine name
        run_machine_uname(str): Run machine uname
        run_driver(str): Run driver
        run_version(str): Version of DB
        run_version_short(str): Shortened version of DB
        label(str): Run label
        source_db_gpu_count(int): Number of GPUs on run machine
        source_db_gpu_driver_ver(str): GPU driver version
        source_db_gpu_name(str): GPU name
        source_db_gpu_mem(str): Amount of GPU mem on run machine
        source_table(str): Table to run query against
        trim(float): Trim decimal to remove from top and bottom of results
        iterations(int): Number of iterations of each query to run
        query_group(str): Query group, usually matches table name
        query_results(dict):::
            query_name(str): Name of query
            query_mapdql(str): Query to run
            query_id(str): Query ID
            query_succeeded(bool): Query succeeded
            query_error_info(str): Query error info
            result_count(int): Number of results returned
            initial_iteration_results(dict):::
                first_execution_time(float): Execution time for first query
                    iteration
                first_connect_time(float):  Connect time for first query
                    iteration
                first_results_iter_time(float): Results iteration time for
                    first query iteration
                first_total_time(float): Total time for first iteration
                first_cpu_mem_usage(float): CPU memory usage for first query
                    iteration
                first_gpu_mem_usage(float): GPU memory usage for first query
                    iteration
            noninitial_iteration_results(list):::
                execution_time(float): Time (in ms) that pymapd reports
                    backend spent on query.
                connect_time(float): Time (in ms) for overhead of query,
                    calculated by subtracting backend execution time from
                    time spent on the execution function.
                results_iter_time(float): Time (in ms) it took to for
                    pymapd.fetchone() to iterate through all of the results.
                total_time(float): Time (in ms) from adding all above times.
            query_total_elapsed_time(int): Total elapsed time for query

      Returns:
        results_dataset(list):::
            result_dataset(dict): Query results dataset
    """
    results_dataset = []
    for query_results in kwargs["queries_results"]:
        if query_results["query_succeeded"]:
            # Aggregate iteration values
            execution_times, connect_times, results_iter_times, total_times = (
                [],
                [],
                [],
                [],
            )
            detailed_timing_last_iteration = {}
            if len(query_results["noninitial_iteration_results"]) == 0:
                # A single query run (most likely a setup or teardown query)
                initial_result = query_results["initial_iteration_results"]
                execution_times.append(initial_result["first_execution_time"])
                connect_times.append(initial_result["first_connect_time"])
                results_iter_times.append(
                    initial_result["first_results_iter_time"]
                )
                total_times.append(initial_result["first_total_time"])
                # Special case
                result_count = 1
            else:
                # More than one query run
                for noninitial_result in query_results[
                    "noninitial_iteration_results"
                ]:
                    execution_times.append(noninitial_result["execution_time"])
                    connect_times.append(noninitial_result["connect_time"])
                    results_iter_times.append(
                        noninitial_result["results_iter_time"]
                    )
                    total_times.append(noninitial_result["total_time"])
                    # Overwrite result count, same for each iteration
                    result_count = noninitial_result["result_count"]

                # If available, getting the last iteration's component-wise timing information as a json structure
                if (
                    query_results["noninitial_iteration_results"][-1]["debug_info"]
                    is not None
                ):
                    detailed_timing_last_iteration = json.loads(
                        query_results["noninitial_iteration_results"][-1][
                            "debug_info"
                        ]
                    )["timer"]
            # Calculate query times
            logging.debug(
                "Calculating times from query " + query_results["query_id"]
            )
            query_times = calculate_query_times(
                total_times=total_times,
                execution_times=execution_times,
                connect_times=connect_times,
                results_iter_times=results_iter_times,
                trim=kwargs[
                    "trim"
                ],  # Trim top and bottom n% for trimmed calculations
            )
            result_dataset = {
                "name": query_results["query_name"],
                "mapdql": query_results["query_mapdql"],
                "succeeded": True,
                "results": {
                    "run_guid": kwargs["run_guid"],
                    "run_timestamp": kwargs["run_timestamp"],
                    "run_connection": kwargs["run_connection"],
                    "run_machine_name": kwargs["run_machine_name"],
                    "run_machine_uname": kwargs["run_machine_uname"],
                    "run_driver": kwargs["run_driver"],
                    "run_version": kwargs["run_version"],
                    "run_version_short": kwargs["run_version_short"],
                    "run_label": kwargs["label"],
                    "run_gpu_count": kwargs["source_db_gpu_count"],
                    "run_gpu_driver_ver": kwargs["source_db_gpu_driver_ver"],
                    "run_gpu_name": kwargs["source_db_gpu_name"],
                    "run_gpu_mem_mb": kwargs["source_db_gpu_mem"],
                    "run_table": kwargs["source_table"],
                    "query_group": kwargs["query_group"],
                    "query_id": query_results["query_id"],
                    "query_result_set_count": result_count,
                    "query_error_info": query_results["query_error_info"],
                    "query_conn_first": query_results[
                        "initial_iteration_results"
                    ]["first_connect_time"],
                    "query_conn_avg": query_times["connect_time_avg"],
                    "query_conn_min": query_times["connect_time_min"],
                    "query_conn_max": query_times["connect_time_max"],
                    "query_conn_85": query_times["connect_time_85"],
                    "query_exec_first": query_results[
                        "initial_iteration_results"
                    ]["first_execution_time"],
                    "query_exec_avg": query_times["execution_time_avg"],
                    "query_exec_min": query_times["execution_time_min"],
                    "query_exec_max": query_times["execution_time_max"],
                    "query_exec_85": query_times["execution_time_85"],
                    "query_exec_25": query_times["execution_time_25"],
                    "query_exec_stdd": query_times["execution_time_std"],
                    "query_exec_trimmed_avg": query_times[
                        "execution_time_trimmed_avg"
                    ],
                    "query_exec_trimmed_max": query_times[
                        "execution_time_trimmed_max"
                    ],
                    # Render queries not supported yet
                    "query_render_first": None,
                    "query_render_avg": None,
                    "query_render_min": None,
                    "query_render_max": None,
                    "query_render_85": None,
                    "query_render_25": None,
                    "query_render_stdd": None,
                    "query_total_first": query_results[
                        "initial_iteration_results"
                    ]["first_total_time"],
                    "query_total_avg": query_times["total_time_avg"],
                    "query_total_min": query_times["total_time_min"],
                    "query_total_max": query_times["total_time_max"],
                    "query_total_85": query_times["total_time_85"],
                    "query_total_all": query_results[
                        "query_total_elapsed_time"
                    ],
                    "query_total_trimmed_avg": query_times[
                        "total_time_trimmed_avg"
                    ],
                    "results_iter_count": kwargs["iterations"],
                    "results_iter_first": query_results[
                        "initial_iteration_results"
                    ]["first_results_iter_time"],
                    "results_iter_avg": query_times["results_iter_time_avg"],
                    "results_iter_min": query_times["results_iter_time_min"],
                    "results_iter_max": query_times["results_iter_time_max"],
                    "results_iter_85": query_times["results_iter_time_85"],
                    "cpu_mem_usage_mb": query_results[
                        "initial_iteration_results"
                    ]["first_cpu_mem_usage"],
                    "gpu_mem_usage_mb": query_results[
                        "initial_iteration_results"
                    ]["first_gpu_mem_usage"],
                },
                "debug": {
                    "query_exec_times": query_times["execution_times"],
                    "query_total_times": query_times["total_times"],
                    "detailed_timing_last_iteration": detailed_timing_last_iteration,
                },
            }
        elif not query_results["query_succeeded"]:
            result_dataset = {
                "name": query_results["query_name"],
                "mapdql": query_results["query_mapdql"],
                "succeeded": False,
            }
        results_dataset.append(result_dataset)
    logging.debug("All values set for query " + query_results["query_id"])
    return results_dataset


def send_results_db(**kwargs):
    """
      Send results dataset to a database using pymapd

      Kwargs:
        results_dataset(list):::
            result_dataset(dict): Query results dataset
        table(str): Results destination table name
        db_user(str): Results destination user name
        db_passwd(str): Results destination password
        db_server(str): Results destination server address
        db_port(int): Results destination server port
        db_name(str): Results destination database name
        table_schema_file(str): Path to destination database schema file

      Returns:
        True(bool): Sending results to destination database succeeded
        False(bool): Sending results to destination database failed. Exception
            should be logged.
    """
    # Create dataframe from list of query results
    logging.debug("Converting results list to pandas dataframe")
    results_df = DataFrame(kwargs["results_dataset"])
    # Establish connection to destination db
    logging.debug("Connecting to destination db")
    dest_con = get_connection(
        db_user=kwargs["db_user"],
        db_passwd=kwargs["db_passwd"],
        db_server=kwargs["db_server"],
        db_port=kwargs["db_port"],
        db_name=kwargs["db_name"],
    )
    if not dest_con:
        logging.exception("Could not connect to destination db.")
        return False
    # Load results into db, creating table if it does not exist
    tables = dest_con.get_tables()
    if kwargs["table"] not in tables:
        logging.info("Destination table does not exist. Creating.")
        try:
            with open(kwargs["table_schema_file"], "r") as table_schema:
                logging.debug(
                    "Reading table_schema_file: " + kwargs["table_schema_file"]
                )
                create_table_sql = table_schema.read().replace("\n", " ")
                create_table_sql = create_table_sql.replace(
                    "##TAB##", kwargs["table"]
                )
        except FileNotFoundError:
            logging.exception("Could not find destination table_schema_file.")
            return False
        try:
            logging.debug("Executing create destination table query")
            dest_con.execute(create_table_sql)
            logging.debug("Destination table created.")
        except (pymapd.exceptions.ProgrammingError, pymapd.exceptions.Error):
            logging.exception("Error running destination table creation")
            return False
    logging.info("Loading results into destination db")
    try:
        dest_con.load_table_columnar(
            kwargs["table"],
            results_df,
            preserve_index=False,
            chunk_size_bytes=0,
            col_names_from_schema=True,
        )
    except (pymapd.exceptions.ProgrammingError, pymapd.exceptions.Error):
        logging.exception("Error loading results into destination db")
        dest_con.close()
        return False
    dest_con.close()
    return True


def send_results_file_json(**kwargs):
    """
      Send results dataset to a local json file

      Kwargs:
        results_dataset_json(str): Json-formatted query results dataset
        output_file_json (str): Location of .json file output

      Returns:
        True(bool): Sending results to json file succeeded
        False(bool): Sending results to json file failed. Exception
            should be logged.
    """
    try:
        logging.debug("Opening json output file for writing")
        with open(kwargs["output_file_json"], "w") as file_json_open:
            logging.info(
                "Writing to output json file: " + kwargs["output_file_json"]
            )
            file_json_open.write(kwargs["results_dataset_json"])
        return True
    except IOError:
        logging.exception("Error writing results to json output file")
        return False


def send_results_jenkins_bench(**kwargs):
    """
      Send results dataset to a local json file formatted for use with jenkins
        benchmark plugin: https://github.com/jenkinsci/benchmark-plugin

      Kwargs:
        results_dataset(list):::
            result_dataset(dict): Query results dataset
        thresholds_name(str): Name to use for Jenkins result field
        thresholds_field(str): Field to use for query threshold in jenkins
        output_tag_jenkins(str): Jenkins benchmark result tag, for different
            sets from same table
        output_file_jenkins (str): Location of .json jenkins file output

      Returns:
        True(bool): Sending results to json file succeeded
        False(bool): Sending results to json file failed. Exception
            should be logged.
    """
    jenkins_bench_results = []
    for result_dataset in kwargs["results_dataset"]:
        logging.debug("Constructing output for jenkins benchmark plugin")
        jenkins_bench_results.append(
            {
                "name": result_dataset["query_id"],
                "description": "",
                "parameters": [],
                "results": [
                    {
                        "name": result_dataset["query_id"]
                        + "_"
                        + kwargs["thresholds_name"],
                        "description": "",
                        "unit": "ms",
                        "dblValue": result_dataset[kwargs["thresholds_field"]],
                    }
                ],
            }
        )
    jenkins_bench_json = json.dumps(
        {
            "groups": [
                {
                    "name": result_dataset["run_table"]
                    + kwargs["output_tag_jenkins"],
                    "description": "Source table: "
                    + result_dataset["run_table"],
                    "tests": jenkins_bench_results,
                }
            ]
        }
    )
    try:
        logging.debug("Opening jenkins_bench json output file for writing")
        with open(kwargs["output_file_jenkins"], "w") as file_jenkins_open:
            logging.info(
                "Writing to jenkins_bench json file: "
                + kwargs["output_file_jenkins"]
            )
            file_jenkins_open.write(jenkins_bench_json)
        return True
    except IOError:
        logging.exception("Error writing results to jenkins json output file")
        return False


def send_results_output(**kwargs):
    """
      Send results dataset script output

      Kwargs:
        results_dataset_json(str): Json-formatted query results dataset

      Returns:
        True(bool): Sending results to output succeeded
    """
    logging.info("Printing query results to output")
    print(kwargs["results_dataset_json"])
    return True


def process_arguments(input_arguments):
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
        "-u",
        "--user",
        dest="user",
        default="mapd",
        help="Source database user",
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
        "-n",
        "--name",
        dest="name",
        default="mapd",
        help="Source database name",
    )
    required.add_argument(
        "-t",
        "--table",
        dest="table",
        required=True,
        help="Source db table name",
    )
    required.add_argument(
        "-l",
        "--label",
        dest="label",
        required=True,
        help="Benchmark run label",
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
        + "Only use when benchmarking against same machine that this script "
        + "is run from.",
    )
    optional.add_argument(
        "-m",
        "--machine-name",
        dest="machine_name",
        help="Name of source machine",
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
        help="Destination table schema file. This must be an executable "
        + "CREATE TABLE statement that matches the output of this script. It "
        + "is required when creating the results table. Default location is "
        + 'in "./results_table_schemas/query-results.sql"',
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
    optional.add_argument(
        "--setup-teardown-queries-dir",
        dest="setup_teardown_queries_dir",
        type=str,
        default=None,
        help='Absolute path to dir with setup & teardown query files. '
        'Query files with "setup" in the filename will be executed in '
        'the setup stage, likewise query files with "teardown" in '
        'the filenname will be executed in the tear-down stage. Queries '
        'execute in lexical order. [Default: None, meaning this option is '
        'not used.]',
    )
    optional.add_argument(
        "--run-setup-teardown-per-query",
        dest="run_setup_teardown_per_query",
        action="store_true",
        help='Run setup & teardown steps per query. '
        'If set, setup-teardown-queries-dir must be specified. '
        'If not set, but setup-teardown-queries-dir is specified '
        'setup & tear-down queries will run globally, that is, '
        'once per script invocation.'
        ' [Default: False]'
    )
    optional.add_argument(
        "-F",
        "--foreign-table-filename",
        dest="foreign_table_filename",
        default=None,
        help="Path to file containing template for import query. "
        "Path must be relative to the FOREIGN SERVER. "
        "Occurances of \"##FILE##\" within setup/teardown queries will be"
        " replaced with this. "
    )
    optional.add_argument(
        "--jenkins-thresholds-name",
        dest="jenkins_thresholds_name",
        default="average",
        help="Name of Jenkins output field.",
    )
    optional.add_argument(
        "--jenkins-thresholds-field",
        dest="jenkins_thresholds_field",
        default="query_exec_trimmed_avg",
        help="Field to report as jenkins output value.",
    )
    args = parser.parse_args(args=input_arguments)
    return args


def benchmark(input_arguments):
    # Set input args to vars
    args = process_arguments(input_arguments)
    verbose = args.verbose
    quiet = args.quiet
    source_db_user = args.user
    source_db_passwd = args.passwd
    source_db_server = args.server
    source_db_port = args.port
    source_db_name = args.name
    source_table = args.table
    label = args.label
    queries_dir = args.queries_dir
    iterations = args.iterations
    gpu_count = args.gpu_count
    gpu_name = args.gpu_name
    no_gather_conn_gpu_info = args.no_gather_conn_gpu_info
    no_gather_nvml_gpu_info = args.no_gather_nvml_gpu_info
    gather_nvml_gpu_info = args.gather_nvml_gpu_info
    machine_name = args.machine_name
    machine_uname = args.machine_uname
    destinations = args.destination
    dest_db_user = args.dest_user
    dest_db_passwd = args.dest_passwd
    dest_db_server = args.dest_server
    dest_db_port = args.dest_port
    dest_db_name = args.dest_name
    dest_table = args.dest_table
    dest_table_schema_file = args.dest_table_schema_file
    output_file_json = args.output_file_json
    output_file_jenkins = args.output_file_jenkins
    output_tag_jenkins = args.output_tag_jenkins
    setup_teardown_queries_dir = args.setup_teardown_queries_dir
    run_setup_teardown_per_query = args.run_setup_teardown_per_query
    foreign_table_filename = args.foreign_table_filename
    jenkins_thresholds_name = args.jenkins_thresholds_name
    jenkins_thresholds_field = args.jenkins_thresholds_field

    # Hard-coded vars
    trim = 0.15

    # Set logging output level
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    elif quiet:
        logging.basicConfig(level=logging.WARNING)
    else:
        logging.basicConfig(level=logging.INFO)

    # Input validation
    if (iterations > 1) is not True:
        # Need > 1 iteration as first iteration is dropped from calculations
        logging.error("Iterations must be greater than 1")
        exit(1)
    if verify_destinations(
        destinations=destinations,
        dest_db_server=dest_db_server,
        output_file_json=output_file_json,
        output_file_jenkins=output_file_jenkins,
    ):
        logging.debug("Destination(s) have been verified.")
    else:
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
    # Set run-specific variables (time, uid, etc.)
    run_vars = get_run_vars(con=con)
    # Set GPU info depending on availability
    gpu_info = get_gpu_info(
        gpu_name=gpu_name,
        no_gather_conn_gpu_info=no_gather_conn_gpu_info,
        con=con,
        conn_machine_name=run_vars["conn_machine_name"],
        no_gather_nvml_gpu_info=no_gather_nvml_gpu_info,
        gather_nvml_gpu_info=gather_nvml_gpu_info,
        gpu_count=gpu_count,
    )
    # Set run machine info
    machine_info = get_machine_info(
        conn_machine_name=run_vars["conn_machine_name"],
        machine_name=machine_name,
        machine_uname=machine_uname,
    )
    # Read queries from files, set to queries dir in PWD if not passed in
    if not queries_dir:
        queries_dir = os.path.join(os.path.dirname(__file__), "queries")
    query_list = read_query_files(
        queries_dir=queries_dir, source_table=source_table
    )
    if not query_list:
        exit(1)
    # Read setup/teardown queries if they exist
    setup_query_list, teardown_query_list =\
        read_setup_teardown_query_files(queries_dir=setup_teardown_queries_dir,
                                        source_table=source_table,
                                        foreign_table_filename=foreign_table_filename)
    # Check at what granularity we want to run setup or teardown queries at
    run_global_setup_queries = setup_query_list is not None and not run_setup_teardown_per_query
    run_per_query_setup_queries = setup_query_list is not None and run_setup_teardown_per_query
    run_global_teardown_queries = teardown_query_list is not None and not run_setup_teardown_per_query
    run_per_query_teardown_queries = teardown_query_list is not None and run_setup_teardown_per_query
    # Run global setup queries if they exist
    queries_results = []
    st_qr = run_setup_teardown_query(queries=setup_query_list,
                                     do_run=run_global_setup_queries, trim=trim, con=con)
    queries_results.extend(st_qr)
    # Run queries
    for query in query_list["queries"]:
        # Run setup queries
        st_qr = run_setup_teardown_query(
            queries=setup_query_list, do_run=run_per_query_setup_queries, trim=trim, con=con)
        queries_results.extend(st_qr)
        # Run benchmark query
        query_result = run_query(
            query=query, iterations=iterations, trim=trim, con=con
        )
        queries_results.append(query_result)
        # Run tear-down queries
        st_qr = run_setup_teardown_query(
            queries=teardown_query_list, do_run=run_per_query_teardown_queries, trim=trim, con=con)
        queries_results.extend(st_qr)
    logging.info("Completed all queries.")
    # Run global tear-down queries if they exist
    st_qr = run_setup_teardown_query(queries=teardown_query_list,
                                     do_run=run_global_teardown_queries, trim=trim, con=con)
    queries_results.extend(st_qr)
    logging.debug("Closing source db connection.")
    con.close()
    # Generate results dataset
    results_dataset = create_results_dataset(
        run_guid=run_vars["run_guid"],
        run_timestamp=run_vars["run_timestamp"],
        run_connection=run_vars["run_connection"],
        run_machine_name=machine_info["run_machine_name"],
        run_machine_uname=machine_info["run_machine_uname"],
        run_driver=run_vars["run_driver"],
        run_version=run_vars["run_version"],
        run_version_short=run_vars["run_version_short"],
        label=label,
        source_db_gpu_count=gpu_info["source_db_gpu_count"],
        source_db_gpu_driver_ver=gpu_info["source_db_gpu_driver_ver"],
        source_db_gpu_name=gpu_info["source_db_gpu_name"],
        source_db_gpu_mem=gpu_info["source_db_gpu_mem"],
        source_table=source_table,
        trim=trim,
        iterations=iterations,
        query_group=query_list["query_group"],
        queries_results=queries_results,
    )
    results_dataset_json = json.dumps(
        results_dataset, default=json_format_handler, indent=2
    )
    successful_results_dataset = [
        x for x in results_dataset if x["succeeded"] is not False
    ]
    successful_results_dataset_results = []
    for results_dataset_entry in successful_results_dataset:
        successful_results_dataset_results.append(
            results_dataset_entry["results"]
        )
    # Send results to destination(s)
    sent_destination = True
    if "mapd_db" in destinations:
        if not send_results_db(
            results_dataset=successful_results_dataset_results,
            table=dest_table,
            db_user=dest_db_user,
            db_passwd=dest_db_passwd,
            db_server=dest_db_server,
            db_port=dest_db_port,
            db_name=dest_db_name,
            table_schema_file=dest_table_schema_file,
        ):
            sent_destination = False
    if "file_json" in destinations:
        if not send_results_file_json(
            results_dataset_json=results_dataset_json,
            output_file_json=output_file_json,
        ):
            sent_destination = False
    if "jenkins_bench" in destinations:
        if not send_results_jenkins_bench(
            results_dataset=successful_results_dataset_results,
            thresholds_name=jenkins_thresholds_name,
            thresholds_field=jenkins_thresholds_field,
            output_tag_jenkins=output_tag_jenkins,
            output_file_jenkins=output_file_jenkins,
        ):
            sent_destination = False
    if "output" in destinations:
        if not send_results_output(results_dataset_json=results_dataset_json):
            sent_destination = False
    if not sent_destination:
        logging.error("Sending results to one or more destinations failed")
        exit(1)
    else:
        logging.info(
            "Succesfully loaded query results info into destination(s)"
        )


if __name__ == "__main__":
    benchmark(sys.argv[1:])
