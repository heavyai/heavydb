# Copyright 2023 HEAVY.AI, Inc.
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

# Connect to conbench PostgreSQL database and print a summary report to STDOUT
# based on a given GIT_COMMIT.
# Requires environment variables (see os.environ calls).
# Exit code is 0, 1, 2 for SUCCESS, UNSTABLE, or FAILURE Jenkins stage status.

import os
import psycopg2
import re

# Parameterize input and output ENV values

# Editable settings
params = {'sha': os.environ['GIT_COMMIT'], # commit to run report on
          'host': os.environ['BENCH_HOST'], # hostname on which to compare statistics
          'n': 17} # number of prior master benchmarks to compare against

benchmark_detail_url = '{}/benchmarks/{{}}/'.format(os.environ['CONBENCH_URL']) # param: benchmark_id

# Highlight without warning any z_scores that fall above 99%.
ZSCORE_CHECK = 2.5758293035489 # statistics.NormalDist().inv_cdf(0.5 + 0.5 * 0.99)
# Warning when any z_score falls above central 99.9% range.
ZSCORE_WARNING = 3.2905267314919255 # statistics.NormalDist().inv_cdf(0.5 + 0.5 * 0.999)
# Error when any benchmark falls above 4 sigma.
ZSCORE_ERROR = 4.0

# Connect to PostgreSQL DB and open a cursor.
conn = psycopg2.connect(os.environ['CONBENCH_DSN'])
cur = conn.cursor()

# Get single-row info for the overall benchmark run on this commit.
cur.execute("""select co.author_avatar, co.author_name, co.branch, co.message, co.repository -- 1 row
  from run r
  join "commit" co on co.id=r.commit_id
  join hardware h on h.id=r.hardware_id
  where h.name=%(host)s and co.sha=%(sha)s
  order by r."timestamp" desc
  limit 1
""", params)

# In python 3 this can/should be replaced w/ import html + html.escape().
def html_escape(s):
    return s.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')

def get_commit_results(cur):
    row = cur.fetchone()
    return { 'author_avatar': row[0],
        'author_name': html_escape(row[1]),
        'branch': html_escape(row[2]),
        'commit_message': html_escape(row[3]),
        'commit_url': '{}/commit/{}'.format(row[4], params['sha']),
        'short_sha': params['sha'][0:8],
        'host': params['host'] }

html_params = get_commit_results(cur)

# Get full report for all benchmark tests on this commit.
# The 1234567.89 is a sentinel value designating a test that failed to run
# that is translated to NULL in the SQL, and None in python.
cur.execute("""with main_run as ( -- 1 row
  select r.id, r."timestamp"
  from run r
  join "commit" co on co.id=r.commit_id
  join hardware h on h.id=r.hardware_id
  where h.name=%(host)s and co.sha=%(sha)s
  order by r."timestamp" desc
  limit 1
), main_stats as ( -- |b| rows. |b| = number of benchmarks (TPC-DS has 99)
  select br.id as benchmark_result_id, ca.name, ca.tags->>'benchmark_name' as benchmark_name,
         nullif(br.mean, 1234567.89) as mean
  from main_run mr
  join benchmark_result br on br.run_id=mr.id
  join "case" ca on ca.id=br.case_id
), prior_runs as ( -- n most recent master runs. n is in the limit clause
  select r.id
  from main_run mr
  join run r on r."timestamp" < mr."timestamp"
  join "commit" co on co.id=r.commit_id
  join hardware h on h.id=r.hardware_id
  where h.name=%(host)s and co.branch='heavyai:master'
  order by r."timestamp" desc
  limit %(n)s
), prior_benchmarks as ( -- n*|b| rows
  select ca.name, ca.tags->>'benchmark_name' as benchmark_name, nullif(br.mean, 1234567.89) as mean
  from prior_runs pr
  join benchmark_result br on br.run_id=pr.id
  join "case" ca on ca.id=br.case_id
), prior_stats as ( -- |b| rows
  select pb.name, pb.benchmark_name, avg(pb.mean) as avg, stddev_samp(pb.mean) as stddev
  from prior_benchmarks pb
  group by pb.name, pb.benchmark_name
)
select ms.benchmark_result_id as benchmark_id, ms.name, ms.benchmark_name, ms.mean as duration_ms, -- |b| rows
       ps.avg, ps.stddev, (ms.mean - ps.avg) / ps.stddev as z_score
from main_stats ms
join prior_stats ps on ps.name=ms.name and ps.benchmark_name=ms.benchmark_name
order by ms.name, ms.benchmark_name
""", params)

header = [desc[0] for desc in cur.description]
rows = cur.fetchall()
# A query is considered "fixed" if it was previously broken, or its z_score significantly improved.
stats = { 'worst_z_score': 0.0, 'nfixes': 0, 'nchecks': 0, 'nwarnings': 0, 'nerrors': 0 }

# Class to format and gather statistics on each row from database.
class Row:
    def __init__(self, header, row):
        self.header = self.fixup_header(header)
        self.row = self.fixup_row(row) if row != None else self.fixup_header(header)
        self.duration_idx = self.header.index('duration_ms')
        self.avg_idx = self.header.index('avg')
        self.z_score_idx = self.header.index('z_score')

    # Conditionally format report cells and accumulate stats on z_scores.
    def cell(self, tag, idx, value):
        if type(value) is str:
            return '<{0}>{1}</{0}>'.format(tag, value)
        else:
            if self.z_score_idx == idx:
                if value == None:
                    if self.row[self.avg_idx] == None:
                        return '<{0}>{1}</{0}>'.format(tag, value)
                    else:
                        stats['nerrors'] += 1
                        return '<{0} class="error">{1}</{0}>'.format(tag, value)
                if stats['worst_z_score'] < value:
                    stats['worst_z_score'] = value
                if value <= -ZSCORE_ERROR:
                    stats['nfixes'] += 1
                    return '<{0} class="fixed">{1:0.3f}</{0}>'.format(tag, value)
                elif value < ZSCORE_CHECK:
                    return '<{0}>{1:0.3f}</{0}>'.format(tag, value)
                elif value < ZSCORE_WARNING:
                    stats['nchecks'] += 1
                    return '<{0} class="check">{1:0.3f}</{0}>'.format(tag, value)
                elif value < ZSCORE_ERROR:
                    stats['nwarnings'] += 1
                    return '<{0} class="warning">{1:0.3f}</{0}>'.format(tag, value)
                else:
                    stats['nerrors'] += 1
                    return '<{0} class="error">{1:0.3f}</{0}>'.format(tag, value)
            else:
                if value == None:
                    return '<{0}>{1}</{0}>'.format(tag, value)
                else:
                    if self.duration_idx == idx and self.row[self.avg_idx] == None:
                        stats['nfixes'] += 1
                        return '<{0} class="fixed">{1:0.3f}</{0}>'.format(tag, value)
                    else:
                        return '<{0}>{1:0.3f}</{0}>'.format(tag, value)

    # Omit columns that are combined with others in the final report.
    def fixup_header(self, header):
        fixup = header[:] # copy, not reference
        fixup.remove('benchmark_id') # Used in benchmark_detail_url
        fixup.remove('benchmark_name') # Column "name" is based on both "name" and "benchmark_name".
        return fixup

    # Omit columns that are combined with others in the final report.
    def fixup_row(self, row_tuple):
        row = list(row_tuple)
        benchmark_id = header.index('benchmark_id')
        name = header.index('name')
        benchmark_name = header.index('benchmark_name')
        # Combine name and benchmark_name together into a single column.
        if row[name] == 'StringDictionaryBenchmark':
            p = re.compile('\w+/(\w+)')
            md = p.search(row[benchmark_name])
            row[name] = 'StringDictionary {}'.format(md.group(1))
        else:
            row[name] = '{} {}'.format(row[name], row[benchmark_name])
        # Hyperlink benchmark name to the conbench page for the specific benchmark.
        row[name] = '<a href="{}">{}</a>'.format(benchmark_detail_url.format(row[benchmark_id]), row[name])
        assert(benchmark_id < benchmark_name) # Must match column removals in fixup_header().
        del row[benchmark_name]
        del row[benchmark_id]
        return row

    # Return the html table row (tr).
    def tr(self, tag):
        zscore_idx = len(self.row) - 1
        return '<tr>{}</tr>'.format(''.join([self.cell(tag,idx,value) for idx, value in enumerate(self.row)]))

def summary_body_rows():
    return "\n".join([
        '<tr><td>Fixed Tests</td><td class="fixed">{nfixes}</td></tr>',
        '<tr><td>Check Tests</td><td class="check">{nchecks}</td></tr>',
        '<tr><td>Warnings</td><td class="warning">{nwarnings}</td></tr>',
        '<tr><td>Errors</td><td class="error">{nerrors}</td></tr>',
        '<tr><td>Worst z_score</td><td>{worst_z_score:0.3f}</td></tr>'
        ]).format(**stats)

# Set html params
html_params['header_row'] = Row(header, None).tr('th')
html_params['body_rows'] = "\n".join([Row(header, row).tr('td') for row in rows])
html_params['summary_body_rows'] = summary_body_rows()

# Print html report
print("""<!DOCTYPE html>
<html>
<head>
  <title>Benchmarks for {branch} / {short_sha} on {host}</title>
  <style>
body {{ font-family: sans-serif }}
table {{ border-collapse: collapse }}
th {{ text-align: right; padding-right: 1em }}
td {{ font-family: monospace; text-align: right; padding-right: 1em }}
td.fixed {{ background-color: LightGreen }}
td.check {{ background-color: Khaki }}
td.warning {{ background-color: Yellow }}
td.error {{ background-color: Red }}
tr:nth-child(even) {{ background-color: LightCyan }}
  </style>
</head>
<body>
<h1>Benchmarks for {branch} / <a href="{commit_url}">{short_sha}</a> on {host}</h1>
<p><a href="{commit_url}">{commit_message}</a></p>
<!-- img disabled due to strict Content Security Policy for HTML Publisher Jenkins plugin -->
<p><!--img alt="avatar" src="{author_avatar}" height=25-->{author_name}</p>
<table>
{summary_body_rows}
</table>
<table>
{header_row}
{body_rows}
</table>
</body>
</html>
""".format(**html_params))

def error_code():
    if stats['nerrors']:
        return 2
    elif stats['nwarnings']:
        return 1
    else:
        return 0

exit(error_code())
