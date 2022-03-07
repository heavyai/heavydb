# Copyright 2022 HEAVY.AI, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import conbench.runner
import json
import os
import subprocess

# Name the derived class after the google benchmark executable.
class GoogleBenchmark:
	'''Run google benchmarks and publish them to the network conbench server.'''

	external = True
	description = 'Google Benchmark'

	def aggregateBenchmarks(self, benchmarks):
		aggregates = {}
		for benchmark in benchmarks:
			if 'aggregate_name' not in benchmark:
				aggregate = aggregates.get(benchmark['name'])
				if aggregate == None:
					aggregates[benchmark['name']] = {
						'data': [ benchmark['real_time'] ],
						'unit': benchmark['time_unit'] }
				else:
					conversion_factor = self.conversionFactor(aggregate['unit'], benchmark['time_unit'])
					aggregate['data'].append(conversion_factor * benchmark['real_time'])
		return aggregates

	def conversionFactor(self, time_unit_to, time_unit_from):
		# See GetTimeUnitString() in https://github.com/google/benchmark/blob/main/include/benchmark/benchmark.h
		powers = { 's': 0, 'ms': 3, 'us': 6, 'ns': 9 }
		return 10 ** (powers[time_unit_to] - powers[time_unit_from])

	# --run-name parameter must be "commit: $GIT_COMMIT" and the current commit must be $GIT_COMMIT.
	def run(self, name, kwargs):
		context = { 'benchmark_language': 'C++' }
		commit = os.environ.get('GIT_COMMIT')
		# Assumes pwd = Benchmarks/conbench and build_dir is build-$GIT_COMMIT
		build_dir = '../../build-{0}'.format(commit)

		benchmark_out = '{0}/{1}-{2}.json'.format(os.getcwd(), name, commit)
		os.system('cd {0}/Tests && ./{1} --benchmark_repetitions=7 --benchmark_out={2}'
			.format(build_dir, name, benchmark_out))
		report = json.load(open(benchmark_out))

		info = report['context']
		info['branch'] = os.environ.get('GIT_BRANCH')
		options = kwargs

		# Aggregate the benchmark_repetitions by benchmark name
		aggregates = self.aggregateBenchmarks(report['benchmarks'])
		for benchmark_name, result in aggregates.items():
			# Different tags correspond to different 'Benchmarks' in conbench
			tags = { 'benchmark_name': benchmark_name }
			yield self.conbench.record(
				result, name, context=context, options=options, output=result, tags=tags, info=info
			)

@conbench.runner.register_benchmark
class StringDictionaryBenchmark(conbench.runner.Benchmark, GoogleBenchmark):
	name = __qualname__
	def run(self, **kwargs):
		yield from GoogleBenchmark.run(self, self.name, kwargs)
