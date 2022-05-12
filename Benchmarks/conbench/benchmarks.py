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
import glob
import os
import re
import subprocess
import time

# Required --run-name "A: B" will display A in the Reason column.
# Recommended example: --run-name "TPC_DS_10GB: $GIT_COMMIT"
# Assumes pwd = Benchmarks/conbench and build_dir is build-$GIT_COMMIT

def conversionFactor(time_unit_to, time_unit_from):
	# See GetTimeUnitString() in https://github.com/google/benchmark/blob/main/include/benchmark/benchmark.h
	powers = { 's': 0, 'ms': 3, 'us': 6, 'ns': 9 }
	return 10 ** (powers[time_unit_to] - powers[time_unit_from])

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
					conversion_factor = conversionFactor(aggregate['unit'], benchmark['time_unit'])
					aggregate['data'].append(conversion_factor * benchmark['real_time'])
		return aggregates

	def run(self, name, kwargs):
		context = { 'benchmark_language': 'C++' }
		commit = os.environ.get('GIT_COMMIT')
		build_dir = '../../build-{0}'.format(commit)
		benchmark_out = '{0}/{1}-{2}.json'.format(os.getcwd(), name, commit)

		command = ['./'+name, '--benchmark_repetitions=7', '--benchmark_out='+benchmark_out]
		subprocess.run(command, cwd=build_dir+'/Tests')

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

class HeavyDbServer:
	def __init__(self, bindir, datadir, port_main, port_http, port_calcite):  # Start heavydb server
		self.initDataDir(bindir, datadir)
		self.heavydb = subprocess.Popen([bindir+'/heavydb', '--allowed-import-paths=["/"]',
			'--allowed-export-paths=["/"]', '--enable-http-binary-server=0', '--port='+str(port_main),
			'--http-port='+str(port_http), '--calcite-port='+str(port_calcite), datadir])
		print('heavydb server started pid='+str(self.heavydb.pid))
	def __del__(self):  # Shutdown heavydb server
		print('Shutting down heavydb server.')
		self.heavydb.terminate()  # Cleaner than kill()
		print('Server return value=%d' % (self.heavydb.wait()))
	def initDataDir(self, bindir, datadir):
		if not os.path.isdir(datadir):
			os.mkdir(datadir, mode=0o775)
			initheavy = subprocess.run([bindir+'/initheavy', datadir])
			assert initheavy.returncode == 0, 'initheavy returned {0}.'.format(initheavy)

# Execute test query to count US states
def numberOfUsStates(bindir, port_main):
	query = b'SELECT COUNT(*) FROM heavyai_us_states;'
	print('Running test query: %s' % (query))
	FAILED_TO_OPEN_TRANSPORT = b'Failed to open transport. Is heavydb running?'
	stdout = FAILED_TO_OPEN_TRANSPORT
	stderr = b''
	attempts = 0
	while stdout.startswith(FAILED_TO_OPEN_TRANSPORT):
		time.sleep(1)
		attempts += 1
		print('Connection attempt {0}'.format(attempts))
		if (120 < attempts):
			print('Too many failed connection attempts. Returning -1.')
			return -1
		heavysql = subprocess.Popen([bindir+'/heavysql', '-p', 'HyperInteractive', '--port', str(port_main),
			'--quiet'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		(stdout, stderr) = heavysql.communicate(query)
	return int(stdout)

def testConnection(bindir, port_main):
	number_of_us_states = numberOfUsStates(bindir, port_main)
	assert number_of_us_states in range(13,100), 'Incorrect number of US states(%d)' % (number_of_us_states)
	print('Counted %d rows in table heavyai_us_states.' % (number_of_us_states))

@conbench.runner.register_benchmark
class StringDictionaryBenchmark(conbench.runner.Benchmark, GoogleBenchmark):
	name = __qualname__
	def run(self, **kwargs):
		yield from GoogleBenchmark.run(self, self.name, kwargs)

@conbench.runner.register_benchmark
class TPC_DS_10GB(conbench.runner.Benchmark):
	'''TPC-DS SQL tests'''

	SCALE='10'
	external = True
	description = 'TPC-DS SCALE=%s SQL tests' % (SCALE)
	name = __qualname__

	BASENAME='TPC-DS_Tools_v3.2.0'
	OUTPUT_DIR='%s_%sGB' % (BASENAME, SCALE)
	PG_TGZ=OUTPUT_DIR + '.tgz'
	DATADIR='storage'
	PORT_MAIN=16274
	PORT_HTTP=16278
	PORT_CALCITE=16279
	SENTINEL_FAILED_TIME=1234567.890 # 20 minutes and 34.56789 seconds is the sentinel failure value

	def checkForRequiredFiles(self):
		assets_dir = os.environ.get('TPCDS_ASSETS_DIR')
		assert assets_dir != None, 'Please set env variable TPCDS_ASSETS_DIR to directory with %s.' % (self.PG_TGZ)
		file = self.PG_TGZ
		assert os.path.exists('%s/.conbench' % os.environ.get('HOME')
		  ), 'A .conbench file is required to submit results to a conbench server.'
		assert os.path.exists('%s/%s' % (assets_dir, file)), 'File %s not found in %s.' % (file, assets_dir)
		return assets_dir

	def setupAndChdirToWorkingDirectory(self, workingdir):
		assets_dir = self.checkForRequiredFiles()
		os.mkdir(workingdir, mode=0o775)
		rakefile = os.path.realpath(os.path.join(os.getcwd(),'../rake/Rakefile'))
		subprocess.run(['ln', '-s', rakefile, workingdir])
		os.chdir(workingdir)
		# Untar postgres results
		subprocess.run(['tar', 'zxf', '%s/%s'%(assets_dir,self.PG_TGZ)])

	def run(self, **kwargs):
		commit = os.environ.get('GIT_COMMIT')
		build_dir = os.path.realpath(os.path.join(os.getcwd(),'../../build-%s'%(commit)))
		self.setupAndChdirToWorkingDirectory(build_dir + '/conbench')
		# Start server on new port
		bindir = build_dir + '/bin'
		heavy_db_server = HeavyDbServer(bindir, self.DATADIR, self.PORT_MAIN, self.PORT_HTTP, self.PORT_CALCITE)
		testConnection(bindir, self.PORT_MAIN)
		# Run rake task
		env = { 'HEAVYSQL': '%s/heavysql -p HyperInteractive --port %d'%(bindir,self.PORT_MAIN)
		      , 'SCALE': self.SCALE, 'SKIP_PG': '1', 'PATH': os.environ.get('PATH') }
		subprocess.run('rake tpcds:compare', env=env, shell=True)
		# report to conbench server
		context = { 'benchmark_language': 'C++' }
		info = { 'branch': os.environ.get('GIT_BRANCH') }
		options = kwargs
		# Read and process output json files
		def query_num(filename):
			md = re.search('/query(\d+).json$', filename)
			return int(md.group(1)) if md else 0
		for filename in sorted(glob.glob(self.OUTPUT_DIR+'/query*.json'), key=query_num):
			with open(filename) as file:
				benchmark = json.load(file)
				tags = { 'benchmark_name': '%02d' % query_num(filename) }
				data = [ benchmark['time_ms'] ] if benchmark['success'] else [ self.SENTINEL_FAILED_TIME ]
				result = { 'data': data, 'unit': 'ms' }
				info['message'] = benchmark['message'][:4096] # truncate large messages
				yield self.conbench.record(
					result, self.name, context=context, options=options, output=result, tags=tags, info=info
				)
