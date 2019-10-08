from run_benchmark import benchmark
import sys
import os
from argparse import ArgumentParser
from synthetic_benchmark.create_table import SyntheticTable
from analyze_benchmark import PrettyPrint, BenchmarkLoader

if __name__ == "__main__":
    # Parse the command line arguments:
    parser = ArgumentParser()
    required = parser.add_argument_group("required arguments")
    required.add_argument("--user", dest="user", default="admin")
    required.add_argument(
        "--password", dest="password", default="HyperInteractive"
    )
    required.add_argument("--name", dest="name", default="omnisci")
    required.add_argument("--server", dest="server", default="localhost")
    required.add_argument("--port", dest="port", default="6274")
    required.add_argument("--dest_user", dest="dest_user", default="admin")
    required.add_argument(
        "--dest_password", dest="dest_password", default="HyperInteractive"
    )
    required.add_argument("--dest_name", dest="dest_name", default="omnisci")
    required.add_argument(
        "--dest_server", dest="dest_server", default="localhost"
    )
    required.add_argument("--dest_port", dest="dest_port", default="6274")
    required.add_argument(
        "--table_name",
        dest="table_name",
        default="omnisci_syn_bench",
        help="Table name to contain all the generated random synthetic data.",
    )
    required.add_argument(
        "--fragment_size",
        dest="fragment_size",
        default="32000000",
        help="Fragment size to be used for the synthetic data on the database",
    )
    required.add_argument(
        "--num_fragments",
        dest="num_fragments",
        default="4",
        help="Number of fragments used to generate synthetic data: "
        + "Total rows in the table: num_fragments * fragment_size.",
    )
    required.add_argument(
        "--data_dir",
        dest="data_dir",
        default=os.getcwd() + "/../build/synthetic_data",
        help="This directory is used (or gets created) to store generated "
        + "random synthetic data (csv files), as well as final results.",
    )
    required.add_argument(
        "--result_dir",
        dest="result_dir",
        default=os.getcwd() + "/../build/synthetic_results",
        help="This directory is used to store results."
        + " Final results are restructured within this directory based on "
        + " the benchmark label and the GPU label.",
    )
    required.add_argument(
        "--query",
        dest="query",
        default="all",
        help="Specifies the the query group to execute particular benchmark queries. "
        + "For example, BaselineHash, MultiStep, NonGroupedAgg, etc."
        + "All query groups can be found at Benchmarks/synthetic_benchmark/queries/ "
        + "The default value is to run all queries (all).",
    )
    required.add_argument(
        "--label",
        dest="label",
        help="This label is used to differentiate different benchmark runs.",
    )
    required.add_argument(
        "--iterations",
        dest="iterations",
        default="2",
        help="Number of iterations used for the benchmark. The first "
        + "iteration will not be included in the final measurements, "
        + "unless specifically asked to report that attribute.",
    )
    required.add_argument(
        "--gpu_count",
        dest="gpu_count",
        default="1",
        help="Number of GPUs used for the benchmark.",
    )
    required.add_argument(
        "--gpu_label",
        dest="gpu_label",
        default="GPU",
        help="This label is used to categorize the stored results (.json) of the benchmark queries. "
        + " Results are stored at {data_dir}/results/{label}/{gpu_label}/Benchmarks/{query}.json",
    )
    required.add_argument(
        "--attribute",
        dest="attribute",
        default="query_exec_avg",
        help="This attribute is used to print out results for each query group. "
        + "Default value is query_total_avg",
    )
    required.add_argument(
        "--skip_data_gen_and_import",
        dest="skip_data_gen_and_import",
        action="store_true",
        help="Skips the data generation, table creation, and import. "
        + "Note that in this case there will be no "
        + "guarantee whether the table exists and data is stored as expected."
        + "It is user's responsibility to make sure everything is in place.",
    )
    required.add_argument(
        "--print_results",
        dest="print_results",
        action="store_true",
        help="If enabled, the results for each particular query group is printed in stdout.",
    )

    args = parser.parse_args()

    assert args.label, "Label is required to store query results."

    # create (or verify existence) of a table with synthetic data in it:
    try:
        if not os.path.isdir(args.data_dir):
            os.makedirs(args.data_dir)
    except OSError:
        print("Failed to create directory %s" % (args.data_dir))

    # create a synthetic table in the database so that benchmark queries can run on them
    if args.skip_data_gen_and_import is False:
        print(" === Preparing the required synthetic data...")
        if args.port != args.dest_port or args.server != args.dest_server:
            is_remote_server = True
        else:
            is_remote_server = False

        try:
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
                is_remote_server=is_remote_server,
            )
            # verify the table existence in the database, if not, creates data,
            # then creates the table, and then import the data to the table
            synthetic_table.createDataAndImportTable()
        except:
            raise Exception("Aborting the benchmark, no valid data found.")
    else:
        print(" === Data generation is skipped...")

    # final results' destination (generated .json files)
    try:
        if not os.path.isdir(args.result_dir):
            os.makedirs(args.result_dir)
    except OSError:
        print("Failed to create directory %s" % (args.result_dir))

    result_dir_name = (
        args.result_dir
        + "/"
        + args.label
        + "/"
        + args.gpu_label
        + "/Benchmarks/"
    )
    try:
        if not os.path.isdir(result_dir_name):
            os.makedirs(result_dir_name)
    except OSError:
        print("Failed to create directory %s" % (result_dir_name))

    query_dir = "synthetic_benchmark/queries/"
    assert os.path.isdir(query_dir)
    all_query_list = os.listdir(query_dir)

    # adjusting the required benchmark arguments with current arguments
    benchmark_args = ["--user", args.user]
    benchmark_args += ["--passwd", args.password]
    benchmark_args += ["--server", args.server]
    benchmark_args += ["--port", args.port]
    benchmark_args += ["--name", args.name]
    benchmark_args += ["--table", args.table_name]
    benchmark_args += ["--iterations", args.iterations]
    benchmark_args += ["--gpu-count", args.gpu_count]
    benchmark_args += ["--destination", "file_json"]
    benchmark_args += ["--dest-user", args.dest_user]
    benchmark_args += ["--dest-passwd", args.dest_password]
    benchmark_args += ["--dest-server", args.dest_server]
    benchmark_args += ["--dest-port", args.dest_port]
    benchmark_args += ["--dest-name", args.dest_name]
    benchmark_args += [
        "--dest-table-schema-file",
        "results_table_schemas/query-results.sql",
    ]
    benchmark_args += ["--label", args.label]
    benchmark_args += [
        "--output-file-json",
        result_dir_name + args.query + ".json",
    ]
    benchmark_args += ["--queries-dir", query_dir + args.query]

    if args.query == "all":
        for query_group in sorted(all_query_list):
            # adjusting query-related args
            benchmark_args[-4:] = [
                "--output-file-json",
                result_dir_name + query_group + ".json",
                "--queries-dir",
                query_dir + query_group,
            ]
            print(" === Running benchmark queries for %s" % (query_group))
            benchmark(benchmark_args)

            if args.print_results:
                refBench = BenchmarkLoader(
                    result_dir_name, os.listdir(result_dir_name)
                )
                refBench.load(query_group + ".json")
                printer = PrettyPrint(
                    refBench, None, args.attribute, False
                ).printAttribute()

    else:
        assert args.query in all_query_list, "Invalid query directory entered,"
        print(" === Running benchmark queries for %s" % (args.query))
        benchmark(benchmark_args)
        if args.print_results:
            refBench = BenchmarkLoader(
                result_dir_name, os.listdir(result_dir_name)
            )
            refBench.load(args.query + ".json")
            printer = PrettyPrint(
                refBench, None, args.attribute, False
            ).printAttribute()
