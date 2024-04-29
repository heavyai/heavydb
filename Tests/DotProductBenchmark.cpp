/*
 * Copyright 2024 HEAVY.AI, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "QueryEngine/ResultSet.h"
#include "QueryRunner/QueryRunner.h"
#include "TestHelpers.h"

#include <absl/strings/str_cat.h>

#include <benchmark/benchmark.h>

using namespace TestHelpers;
using QR = QueryRunner::QueryRunner;

constexpr bool hoist_literals = true;
constexpr bool allow_loop_joins = false;
constexpr bool translate_strings = true;
constexpr bool decimal_to_double = true;

std::once_flag setup_flag;
void global_setup() {
  TestHelpers::init_logger_stderr_only();
  QR::init(BASE_PATH);
}

void run_ddl_statement(const std::string& create_table_stmt) {
  QR::get()->runDDLStatement(create_table_stmt);
}

std::shared_ptr<ResultSet> run_multiple_agg(const std::string& query_str,
                                            const ExecutorDeviceType device_type) {
  return QR::get()->runSQL(query_str, device_type, hoist_literals, allow_loop_joins);
}

TargetValue run_simple_agg(const std::string& query_str,
                           const ExecutorDeviceType device_type) {
  auto rows = QR::get()->runSQL(query_str, device_type, hoist_literals, allow_loop_joins);
  auto crt_row = rows->getNextRow(translate_strings, decimal_to_double);
  CHECK_EQ(1u, crt_row.size()) << query_str;
  return crt_row[0];
}

/// Represent the array literal: {first, 1.0, 2.0, 3.0, ..., size-1}
struct ArrayLiteral {
  size_t first;
  size_t size;
};

std::ostream& operator<<(std::ostream& os, ArrayLiteral const array_literal) {
  if (array_literal.size) {
    os << '{' << array_literal.first << ".0";
    for (size_t i = 1u; i < array_literal.size; ++i) {
      os << ',' << i << ".0";
    }
    os << '}';
  } else {
    os << "{}";
  }
  return os;
}

class DotProduct : public benchmark::Fixture {
 public:
  void SetUp(const ::benchmark::State& state) override {
    std::call_once(setup_flag, global_setup);
    auto const num_rows = static_cast<size_t>(state.range(0));
    auto const array_size = static_cast<size_t>(state.range(1));
    // Use static variables because when num_rows is increased from 4M to 8M
    // with the same array_size, we can append to the existing table.
    static size_t last_num_rows{};
    static size_t last_array_size{};
    if (last_array_size != array_size) {
      // When the array_size changes, we have to recreate the table.
      run_ddl_statement("DROP TABLE IF EXISTS dot_product_bench;");
      run_ddl_statement(
          absl::StrCat("CREATE TABLE dot_product_bench (c FLOAT[", array_size, "]);"));
      last_array_size = array_size;
      last_num_rows = 0u;
    }
    // Import from CSV because large num_rows INSERT statements take too long.
    TempFile csv;
    for (size_t i = last_num_rows; i < num_rows; ++i) {
      // Only the first array element changes so that we don't have all duplicate rows,
      // in case there's a caching effect.  This is multiplied by 0 in the literal
      // array so doesn't change the final DOT_PRODUCT() value.
      csv << ArrayLiteral{i, array_size} << '\n';
    }
    csv.close();
    run_ddl_statement(absl::StrCat(
        "COPY dot_product_bench FROM '", csv.path(), "' WITH (header='false');"));
    last_num_rows = num_rows;
  }
};

/////////////////////////////////////////////////////////////////////

#define CHECK_FOR_CORRECT_RESULT 1

BENCHMARK_DEFINE_F(DotProduct, NumRowsVsLiteralArraySize)(benchmark::State& state) {
#if CHECK_FOR_CORRECT_RESULT
  auto const num_rows = static_cast<size_t>(state.range(0));
#endif
  auto const array_size = static_cast<size_t>(state.range(1));
  std::ostringstream query;
  query << "SELECT DOT_PRODUCT(c, " << ArrayLiteral{0u, array_size}
        << ") FROM dot_product_bench;";
  std::string const select = query.str();
#if CHECK_FOR_CORRECT_RESULT
  auto const expected =
      static_cast<float>((array_size - 1u) * array_size * (2u * array_size - 1u) / 6u);
#endif
  for (auto _ : state) {
#if CHECK_FOR_CORRECT_RESULT
    auto result_set = run_multiple_agg(select, ExecutorDeviceType::GPU);
    EXPECT_EQ(num_rows, result_set->rowCount());
    // Though there are multiple rows returned, just check the first row.
    auto actual = result_set->getNextRow(translate_strings, decimal_to_double)[0];
    constexpr float EPS = 1e-5;
    EXPECT_NEAR(expected, v<float>(actual), EPS * expected);
#else
    run_multiple_agg(select, ExecutorDeviceType::GPU);
#endif
  }
}

/* Run with --benchmark_output=dot_product.json then run
 * python script to plot data from json output in Mathematica:
import json
from collections import defaultdict

# Load JSON
with open("dot_product.json", "r") as file:
    json_data = json.load(file)

# Process json_data
plot_data = defaultdict(list)
for benchmark in json_data["benchmarks"]:
    parts = benchmark["name"].split('/')
    num_rows = int(parts[2])
    array_size = int(parts[3])
    real_time = benchmark["real_time"]
    # Group by array_size
    plot_data[array_size].append((num_rows, real_time))

# Prepare output
plot_data_list = [list(value) for key, value in sorted(plot_data.items())]
plot_legends = sorted(plot_data.keys())

def format_for_mathematica(data):
    """Convert Python data structures to Mathematica's syntax."""
    formatted_data = str(data)
    formatted_data = formatted_data.replace('[', '{').replace(']', '}').replace('(',
'{').replace(')', '}') return formatted_data

# Print Mathematica instructions
print(f"plotData = {format_for_mathematica(plot_data_list)}")
print(f"plotLegends = {format_for_mathematica(plot_legends)}")
print("StackedListPlot[plotData, PlotLegends -> plotLegends]")
 */

// Configure two benchmarks with exponential growth for two parameters
BENCHMARK_REGISTER_F(DotProduct, NumRowsVsLiteralArraySize)
    ->RangeMultiplier(2)
    ->Ranges({/*num_rows=*/{1ul << 13, 1ul << 23}, /*array_size=*/{2, 1024}})
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
