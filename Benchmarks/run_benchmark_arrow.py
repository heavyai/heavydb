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
from run_benchmark import (
    verify_destinations,
    get_connection,
    get_run_vars,
    get_gpu_info,
    get_machine_info,
    read_query_files,
    validate_query_file,
    get_mem_usage,
    json_format_handler,
    send_results_db,
    send_results_file_json,
    send_results_jenkins_bench,
    send_results_output,
)
from argparse import ArgumentParser


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
          arrow_conversion_time(float): Time (in ms) for converting and
                                    serializing results in arrow format
          total_time(float): Time (in ms) from adding all above times.
        False(bool): The query failed. Exception should be logged.
    """
    start_time = timeit.default_timer()
    query_result = {}
    arrow_cpu_output = kwargs["arrow_cpu_output"]
    try:
        # Run the query
        if arrow_cpu_output:
            query_result = kwargs["con"].select_ipc(kwargs["query_mapdql"])
        else:
            query_result = kwargs["con"]._client.sql_execute_gdf(
                kwargs["con"]._session,
                kwargs["query_mapdql"],
                device_id=0,
                first_n=-1,
            )
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
    execution_time = 0
    if arrow_cpu_output:
        execution_time = query_result._tdf.execution_time_ms
    else:
        execution_time = query_result.execution_time_ms
    connect_time = round((query_elapsed_time - execution_time), 1)
    arrow_conversion_time = 0
    if arrow_cpu_output:
        arrow_conversion_time = query_result._tdf.arrow_conversion_time_ms
    else:
        arrow_conversion_time = query_result.arrow_conversion_time_ms
    # Iterate through each result from the query
    logging.debug(
        "Counting results from query"
        + kwargs["query_name"]
        + " iteration "
        + str(kwargs["iteration"])
    )
    result_count = 0
    start_time = timeit.default_timer()
    if arrow_cpu_output:
        result_count = len(query_result.index)
    # TODO(Wamsi): Add support for computing cuDF size, once cuDF is fixed.
    query_execution = {
        "result_count": result_count,
        "execution_time": execution_time,
        "connect_time": connect_time,
        "arrow_conversion_time": arrow_conversion_time,
        "total_time": execution_time
        + connect_time
        + arrow_conversion_time,
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
        connect_times(list): List of connect_time calculations
        arrow_conversion_times(list): List of arrow_conversion_time calculations
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
        "arrow_conversion_time_avg": round(
            numpy.mean(kwargs["arrow_conversion_times"]), 1
        ),
        "arrow_conversion_time_min": round(
            numpy.min(kwargs["arrow_conversion_times"]), 1
        ),
        "arrow_conversion_time_max": round(
            numpy.max(kwargs["arrow_conversion_times"]), 1
        ),
        "arrow_conversion_time_85": round(
            numpy.percentile(kwargs["arrow_conversion_times"], 85), 1
        ),
        "arrow_conversion_time_25": round(
            numpy.percentile(kwargs["arrow_conversion_times"], 25), 1
        ),
        "arrow_conversion_time_std": round(
            numpy.std(kwargs["arrow_conversion_times"]), 1
        ),
    }

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
            arrow_cpu_output=kwargs["arrow_cpu_output"],
            
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
                first_total_time = (
                    first_execution_time
                    + first_connect_time
                )
                query_results.update(
                    initial_iteration_results={
                        "first_execution_time": first_execution_time,
                        "first_connect_time": first_connect_time,
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
                arrow_conversion_time(float): Time (in ms) it took for
                    arrow conversion and serialization fo results.
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
            (
                execution_times,
                connect_times,
                arrow_conversion_times,
                total_times,
            ) = (
                [],
                [],
                [],
                [],
            )
            for noninitial_result in query_results[
                "noninitial_iteration_results"
            ]:
                execution_times.append(noninitial_result["execution_time"])
                connect_times.append(noninitial_result["connect_time"])
                arrow_conversion_times.append(noninitial_result["arrow_conversion_time"]
                )
                total_times.append(noninitial_result["total_time"])
                # Overwrite result count, same for each iteration
                result_count = noninitial_result["result_count"]
            # Calculate query times
            logging.debug(
                "Calculating times from query " + query_results["query_id"]
            )
            query_times = calculate_query_times(
                total_times=total_times,
                execution_times=execution_times,
                connect_times=connect_times,
                arrow_conversion_times=arrow_conversion_times,
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
                    "query_arrow_conversion_avg": query_times[
                        "arrow_conversion_time_avg"
                    ],
                    "query_arrow_conversion_min": query_times[
                        "arrow_conversion_time_min"
                    ],
                    "query_arrow_conversion_max": query_times[
                        "arrow_conversion_time_max"
                    ],
                    "query_arrow_conversion_85": query_times[
                        "arrow_conversion_time_85"
                    ],
                    "query_arrow_conversion_25": query_times[
                        "arrow_conversion_time_25"
                    ],
                    "query_arrow_conversion_stdd": query_times[
                        "arrow_conversion_time_std"
                    ],
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
        default="results_arrow",
        help="Destination mapd_db table name",
    )
    optional.add_argument(
        "-C",
        "--dest-table-schema-file",
        dest="dest_table_schema_file",
        default="results_table_schemas/arrow-results.sql",
        help="Destination table schema file. This must be an executable "
        + "CREATE TABLE statement that matches the output of this script. It "
        + "is required when creating the results_arrow table. Default location is "
        + 'in "./results_table_schemas/arrow-results.sql"',
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
        "--enable-arrow-cpu-output",
        dest="arrow_cpu_output",
        action="store_true",
        help="Output results in Apache Arrow Serialized format on CPU",
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
    arrow_cpu_output = args.arrow_cpu_output

    # Hard-coded vars
    trim = 0.15
    jenkins_thresholds_name = "average"
    jenkins_thresholds_field = "query_exec_avg"

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
    # Run queries
    queries_results = []
    for query in query_list["queries"]:
        query_result = run_query(
            query=query,
            iterations=iterations,
            trim=trim,
            con=con,
            arrow_cpu_output=arrow_cpu_output,
        )
        queries_results.append(query_result)
    logging.info("Completed all queries.")
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
