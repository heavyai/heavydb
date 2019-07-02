import json
import sys
from os import listdir
import os.path
import getopt

# loads a single benchmark json at a time, and can access its data


class BenchmarkLoader:
    def __init__(self, dir_name, filename_list):
        self.dir_name = dir_name
        self.filename_list = filename_list
        self.data = []

    # reading the benchmark json file
    def load(self, bench_filename):
        assert bench_filename in self.filename_list

        with open(self.dir_name + bench_filename) as json_file:
            self.data = json.load(json_file)

    def getFrontAttribute(self, attribute):
        if self.data:
            return self.data[0]["results"][attribute]
        else:
            return "None"

    def getExistingDataRunLabel(self):
        return self.getFrontAttribute("run_label")

    def getGpuName(self):
        return self.getFrontAttribute("run_gpu_name")

    def getRunTableName(self):
        return self.getFrontAttribute("run_table")

    # return a list of the attribute stored in self.data
    def fetchAttribute(self, attribute):
        result = []
        for experiment in self.data:
            result.append(experiment["results"][attribute])
        return result

    def fetchQueryNames(self):
        result = []
        for experiment in self.data:
            result.append(experiment["results"]["query_id"])
        return result


class BenchAnalyzer:
    def __init__(self, ref, sample, attribute):
        assert isinstance(ref, BenchmarkLoader)
        assert isinstance(sample, BenchmarkLoader)
        self.__header_info = [ref.getRunTableName(), attribute]
        self.__label_name_ref = ref.fetchQueryNames()
        self.__label_name_sample = sample.fetchQueryNames()
        assert self.__label_name_ref == self.__label_name_sample
        self.__attribute_ref = ref.fetchAttribute(attribute)
        self.__attribute_sample = sample.fetchAttribute(attribute)

    def printHeader(self):
        for h in self.__header_info:
            print("  " + h, end="")

    def findAnomaliesRatio(self, epsilon):
        found = False
        speedup = compute_speedup(
            self.__attribute_ref, self.__attribute_sample
        )
        print("Differences outside of %2.0f%%: " % (epsilon * 100), end="")
        self.printHeader()
        for i in range(len(speedup)):
            if abs(speedup[i] - 1.0) > epsilon:
                if found == False:
                    found = True
                print(
                    "\n%s: reference = %.2f ms, sample = %.2f ms, speedup = %.2fx"
                    % (
                        self.__label_name_ref[i],
                        self.__attribute_ref[i],
                        self.__attribute_sample[i],
                        speedup[i],
                    ),
                    end="",
                )
        if found == False:
            print(": None", end="")
        print(
            "\n======================================================================="
        )


def compute_speedup(x, y):
    result = []
    zipped = list(zip(x, y))
    for q in zipped:
        result.append(q[0] / q[1])
    return result


class PrettyPrint:
    def __init__(self, ref, sample, attribute, num_items_per_line=5):
        assert isinstance(ref, BenchmarkLoader)
        assert isinstance(sample, BenchmarkLoader)
        self.__header_info = [
            ref.getRunTableName(),
            attribute,
            ref.getGpuName(),
        ]
        self.__num_items_per_line = num_items_per_line
        self.__label_name_ref = ref.fetchQueryNames()
        self.__label_name_sample = sample.fetchQueryNames()
        assert self.__label_name_ref == self.__label_name_sample
        self.__attribute_ref = ref.fetchAttribute(attribute)
        self.__attribute_sample = sample.fetchAttribute(attribute)
        self.__ref_line_count = 0
        self.__sample_line_count = 0

    def printSolidLine(self, pattern):
        for i in range(self.__num_items_per_line + 1):
            for j in range(11):
                print(pattern, end="")
        print("")

    def printHeader(self):
        for h in self.__header_info:
            print("\t" + h)
        self.printSolidLine("=")

    def getRefElementsPerLine(self):
        return self.__ref_line_count * self.__num_items_per_line

    def printLine(self, array):
        begin = self.getRefElementsPerLine()
        end = self.getRefElementsPerLine() + self.__num_items_per_line
        for i in range(begin, min(end, len(self.__attribute_ref))):
            if isinstance(array[i], float):
                print("%10.2f" % (array[i]), end="")
            elif isinstance(array[i], str):
                print("%10s" % (array[i]), end="")
            else:
                assert False
        print("")

    def printAttribute(self):
        self.printHeader()
        ref_count = len(self.__attribute_ref)
        while self.getRefElementsPerLine() < ref_count:
            print("%10s" % "Queries", end="")
            self.printLine(self.__label_name_ref)
            self.printSolidLine("-")
            print("%10s" % "Reference", end="")
            self.printLine(self.__attribute_ref)
            print("%10s" % "Sample", end="")
            self.printLine(self.__attribute_sample)
            print("%10s" % "Speedup", end="")
            self.printLine(
                compute_speedup(self.__attribute_ref, self.__attribute_sample)
            )
            self.printSolidLine("=")
            self.__ref_line_count += 1
        print("\n\n\n")


def main(argv):
    try:
        opts, args = getopt.getopt(
            argv,
            "hs:r:e:p",
            ["help", "sample=", "reference=", "epsilon=", "print"],
        )
    except getopt.GetOptError:
        print(
            "python3 analyze-benchmark.py -s <sample dir> -r <reference dir> -e <epsilon> -p"
        )
        sys.exit(2)

    dir_artifact_sample = ""
    dir_artifact_ref = ""
    epsilon = 0.05
    to_print = False  # printing all the results, disabled by default

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(
                """
    -s/--sample:\t\t\t directory of the results for the benchmarked sample branch
    -r/--reference:\t\t\t directory of the results for the benchmarked reference branch 
    -e/--epsilon:\t\t\t ratio tolerance for reporting results outside this range
    -p/--print:\t\t\t\t print all the results  
                """
            )
            sys.exit()
        else:
            if opt in ("-s", "--sample"):
                dir_artifact_sample = arg
                assert os.path.isdir(dir_artifact_sample)
            elif opt in ("-r", "--reference"):
                dir_artifact_ref = arg
                assert os.path.isdir(dir_artifact_ref)
            elif opt in ("-e", "--epsilon"):
                epsilon = float(arg)
            elif opt in ("-p", "--print"):
                to_print = True

    assert dir_artifact_ref is not ""
    assert dir_artifact_sample is not ""
    assert epsilon <= 1

    GPU_list_ref = listdir(dir_artifact_ref)
    GPU_list_sample = listdir(dir_artifact_sample)

    for gpu in GPU_list_ref:
        dir_name_ref = dir_artifact_ref + "/" + gpu + "/Benchmarks"
        filename_list_ref = listdir(dir_name_ref)
        dir_name_ref += "/"

        refBench = BenchmarkLoader(dir_name_ref, filename_list_ref)

        if gpu in GPU_list_sample:
            dir_name_sample = dir_artifact_sample + "/" + gpu + "/Benchmarks"
            filename_list_sample = listdir(dir_name_sample)
            dir_name_sample += "/"

            sampleBench = BenchmarkLoader(
                dir_name_sample, filename_list_sample
            )
            first_header = True
            for index in range(len(filename_list_ref)):
                refBench.load(filename_list_ref[index])
                if filename_list_ref[index] in filename_list_sample:
                    sampleBench.load(filename_list_ref[index])
                    if first_header:
                        print(
                            "\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
                        )
                        print("++++ " + sampleBench.getGpuName())
                        print(
                            "++++ reference("
                            + refBench.getFrontAttribute("run_label")
                            + "): "
                            + refBench.getFrontAttribute("run_version")
                        )
                        print(
                            "++++ sample("
                            + sampleBench.getFrontAttribute("run_label")
                            + "): "
                            + sampleBench.getFrontAttribute("run_version")
                        )
                        print(
                            "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
                        )
                        first_header = False

                    analyzer = BenchAnalyzer(
                        refBench, sampleBench, "query_total_avg"
                    )
                    analyzer.findAnomaliesRatio(epsilon)
                    if to_print:
                        printer = PrettyPrint(
                            refBench, sampleBench, "query_total_avg"
                        )
                        printer.printAttribute()
                else:
                    print(
                        "No sample results for table "
                        + refBench.getRunTableName()
                        + " were found."
                    )
                    print(
                        "======================================================================="
                    )

        else:
            print(
                "\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
            )
            print("++++ No sample results for GPU " + gpu + " were found.")
            print(
                "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
            )


if __name__ == "__main__":
    main(sys.argv[1:])
