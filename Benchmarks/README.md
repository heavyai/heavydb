# MapD Benchmark Script

Python script that leverages [pymapd](https://github.com/omnisci/pymapd) to query a mapd database and reports performance of the queries.

## Setup

Script is intended for use on Python3

All required python packages can be installed with `pip` using the `requirements.txt` file:
```
pip install -r requirements.txt
```

The required packages can otherwise be installed individually using [conda](https://conda.io/) or [pip](https://pypi.org/project/pip/). List of required packages:

1) [pymapd](https://github.com/omnisci/pymapd) - Provides a python DB API 2.0-compliant OmniSci interface (formerly MapD)

   Install with conda: `conda install -c conda-forge pymapd` or with pip: `pip install pymapd`

2) [pandas](https://pandas.pydata.org/) - Provides high-performance, easy-to-use data structures and data analysis tools

   Install with conda: `conda install pandas` or with pip: `pip install pandas`

3) [numpy](http://www.numpy.org/) - Package for scientific computing with Python

   Install with conda: `conda install -c anaconda numpy` or with pip: `pip install numpy`

## Usage

### Required components

Running the script requies a few components:

1) Connection to a mapd db with a dataset loaded - either a large sample dataset used for benchmarking, or a custom dataset.
2) Query(ies) to run against the dataset. These can be provided as files with the following syntax: `query_<query_id>.sql` (where query_id is a unique query ID). The script will find all queries from files that match the syntax in the directory passed in to the script at runtime.
3) Destination. Depending on the type of destination, a connection to a mapd db, or destination file location may be required.

### Running the script

Usage can be printed at any time by running: `./run-benchmark.py -h` or `--help`

Currently, usage is:

```
usage: run-benchmark.py [-h] [-v] [-q] [-u USER] [-p PASSWD] [-s SERVER]
                        [-o PORT] [-n NAME] -t TABLE -l LABEL [-d QUERIES_DIR]
                        -i ITERATIONS -g GPUS [-e DESTINATION] [-U DEST_USER]
                        [-P DEST_PASSWD] [-S DEST_SERVER] [-O DEST_PORT]
                        [-N DEST_NAME] [-T DEST_TABLE] [-j OUTPUT_FILE_JSON]

required arguments:
  -u USER, --user USER  Source database user
  -p PASSWD, --passwd PASSWD
                        Source database password
  -s SERVER, --server SERVER
                        Source database server hostname
  -n NAME, --name NAME  Source database name
  -t TABLE, --table TABLE
                        Source db table name
  -l LABEL, --label LABEL
                        Benchmark run label
  -d QUERIES_DIR, --queries-dir QUERIES_DIR
                        Absolute path to dir with query files. [Default:
                        "queries" dir in same location as script]
  -i ITERATIONS, --iterations ITERATIONS
                        Number of iterations per query. Must be > 1
  -g GPUS, --gpus GPUS  Number of GPUs

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         Turn on debug logging
  -q, --quiet           Suppress script outuput (except warnings and errors)
  -o PORT, --port PORT  Source database server port
  -e DESTINATION, --destination DESTINATION
                        Destination type: [mapd_db, file_json, output]
                        Multiple values can be input seperated by commas, ex:
                        "mapd_db,file_json"
  -U DEST_USER, --dest-user DEST_USER
                        Destination mapd_db database user
  -P DEST_PASSWD, --dest-passwd DEST_PASSWD
                        Destination mapd_db database password
  -S DEST_SERVER, --dest-server DEST_SERVER
                        Destination mapd_db database server hostname (required
                        if destination = "mapd_db")
  -O DEST_PORT, --dest-port DEST_PORT
                        Destination mapd_db database server port
  -N DEST_NAME, --dest-name DEST_NAME
                        Destination mapd_db database name
  -T DEST_TABLE, --dest-table DEST_TABLE
                        Destination mapd_db table name
  -j OUTPUT_FILE_JSON, --output-file-json OUTPUT_FILE_JSON
                        Absolute path of .json output file (required if
                        destination = "file_json")
```

Example 1:
```
python ./run-benchmark.py -t flights_2008_10k -l TestLabel -d /data/queries/flights -i 10 -g 4 -S localhost
```
this would run the script with the following parameters:
- Default values for source mapd db: localhost, default username and password, and database name
- Queries would be run against the "flights_2008_10k" table
- Results would have label "TestLabel"
- Query file(s) would be sources from directory "/data/queries/flights"
- Query(ies) would run for 10 iterations each
- Results would show that mapd_db machine has 4 GPUs
- Destination mapd db is located at localhost with the default username, password, and database name.

Example 2:
```
python ./run-benchmark.py -u user -p password -s mapd-server.example.com -n mapd_db -t flights_2008_10k -l TestLabel -d /home/mapd/queries/flights -i 10 -g 4 -e mapd_db,file_json -U dest_user -P password -S mapd-dest-server.mapd.com -N mapd_dest_db -T benchmark_results -j /home/mapd/benchmark_results/example.json
```
