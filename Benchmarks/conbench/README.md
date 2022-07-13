# Conbench Client

`benchmarks.py` is a conbench client script for running and publishing benchmarks to a running conbench server.

You don't need to run this manually. It should be taken care of by Jenkins. The below instructions are in case you would like to run the client elsewhere.

## Requirements

* Install [conbench](https://pypi.org/project/conbench/) via `pip install conbench`.

## Usage

* Run `conbench --help` from the directory with the `benchmarks.py` script. The output should look something like
```
$ conbench --help
Usage: conbench [OPTIONS] COMMAND [ARGS]...

  Conbench: Language-independent Continuous Benchmarking (CB) Framework

Options:
  --help  Show this message and exit.

Commands:
  StringDictionaryBenchmark  Google Benchmark
  list                       List of benchmarks (for orchestration).
  version                    Display Conbench version.
```
* Create a `.conbench` file in the same directory with contents:
```
url: http://10.0.0.37:5000/
email: YOUR_EMAIL
password: YOUR_PASSWORD
```
You may sign up for an account at http://10.0.0.37:5000/ .
* The test corresponding to the benchmark name, e.g. `StringDictionaryBenchmark`, must be compiled and runnable in the current repository.
* The current git commit, as output from `git show -s --format=%H`, is the commit that will be associated with the recorded benchmarks.  **This commit must be known by github.**  Do not run `conbench` on an unpublished commit, otherwise `conbench` will not be able to obtain necessary data on the commit and will result in errors in the reporting.
* The build\_dir is assumed to be `build-$GIT_COMMIT`.
It will `cd` to its `Tests` subdirectory, run the benchmark, capture and parse the output, and report the results to the conbench server.
* View and analyze the results at http://10.0.0.37:5000/.
