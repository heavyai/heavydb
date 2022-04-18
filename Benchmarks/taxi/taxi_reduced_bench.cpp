#include <benchmark/benchmark.h>

#include "Tests/ArrowSQLRunner/ArrowSQLRunner.h"

#include <boost/filesystem/path.hpp>
#include <boost/program_options.hpp>

extern bool g_enable_heterogeneous_execution;
extern bool g_enable_multifrag_heterogeneous_execution;
boost::filesystem::path g_data_path;

using namespace TestHelpers::ArrowSQLRunner;

static void createTaxiReducedTable() {
  ArrowStorage::TableOptions TO{1'000'000};
  createTable("trips",
              {{"pickup_datetime", SQLTypeInfo(kTIMESTAMP, 0, 0)},
               {"passenger_count", SQLTypeInfo(kSMALLINT)},
               {"trip_distance", SQLTypeInfo(kDECIMAL, 14, 2)},
               {"total_amount", SQLTypeInfo(kDECIMAL, 14, 2)},
               {"cab_type", SQLTypeInfo(kVARCHAR, true, kENCODING_DICT)}},
              TO);
}
static void populateTaxiReducedTable() {
  namespace fs = boost::filesystem;
  if (fs::is_directory(g_data_path)) {
    throw std::runtime_error("Directories are not supported.");
  } else {
    std::ifstream is(g_data_path);
    std::string line;
    while (std::getline(is, line)) {
      insertCsvValues("trips", line);
    }
  }
}

template <class T>
T v(const TargetValue& r) {
  auto scalar_r = boost::get<ScalarTargetValue>(&r);
  auto p = boost::get<T>(scalar_r);
  return *p;
}

static void taxi_q1(benchmark::State& state) {
  auto res =
      v<int64_t>(run_simple_agg("select count(*) from trips", ExecutorDeviceType::CPU));
  std::cerr << "number of loaded tuples: " << res << std::endl;
  for (auto _ : state) {
    run_multiple_agg("select cab_type, count(*) from trips group by cab_type",
                     ExecutorDeviceType::CPU);
  }
}

BENCHMARK(taxi_q1);

int main(int argc, char* argv[]) {
  ::benchmark::Initialize(&argc, argv);

  namespace po = boost::program_options;
  namespace fs = boost::filesystem;

  po::options_description desc("Options");
  desc.add_options()("enable-heterogeneous",
                     po::value<bool>(&g_enable_heterogeneous_execution)
                         ->default_value(g_enable_heterogeneous_execution)
                         ->implicit_value(true),
                     "Allow heterogeneous execution.");
  desc.add_options()("enable-multifrag",
                     po::value<bool>(&g_enable_multifrag_heterogeneous_execution)
                         ->default_value(g_enable_multifrag_heterogeneous_execution)
                         ->implicit_value(true),
                     "Allow multifrag heterogeneous execution.");
  desc.add_options()("data", po::value<fs::path>(&g_data_path), "Path to taxi dataset.");

  logger::LogOptions log_options(argv[0]);
  log_options.severity_ = logger::Severity::FATAL;
  log_options.set_options();  // update default values
  desc.add(log_options.get_options());

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << "Usage:" << std::endl << desc << std::endl;
  }

  logger::init(log_options);
  init();

  try {
    createTaxiReducedTable();
    populateTaxiReducedTable();
    ::benchmark::RunSpecifiedBenchmarks();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
    return -1;
  }

  reset();
}