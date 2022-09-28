#!/usr/bin/perl

# Copyright 2022 HEAVY.AI, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Usage: perl filter_by_request_id.pl [request_id] < log/heavydb.INFO

# If request_id is given then print only log files with that request_id,
#    and all lines that follow that share a common root request_id.
# If request_id is not given then print all log lines.
# In both cases, the root request_id is inserted as a new 4th field.

$request_id = shift if 0 < @ARGV && $ARGV[0] =~ /^\d+$/;

while (<>) {
  if (/^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{6} \w \d+) (\d+) (.+)$/) {
    my ($prefix, $rid, $suffix) = ($1, $2, $3);
    if (/This request has parent request_id\((\d+)\)$/) {
      $map{$rid} = exists($map{$1}) ? $map{$1} : $1  # map : $rid -> $root_rid
    }
    my $root_rid = exists($map{$rid}) ? $map{$rid} : $rid;
    $print_lines = !defined($request_id) || $request_id eq $root_rid || $request_id eq $rid;
    print "$prefix $root_rid $rid $suffix\n" if $print_lines
  } elsif ($print_lines) {
    print
  }
}

if (defined $request_id and exists $map{$request_id}) {
  print "\nRe-run with root request_id = $map{$request_id} to get additional related log lines.\n"
}
