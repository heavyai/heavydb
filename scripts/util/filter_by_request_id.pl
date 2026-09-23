#!/usr/bin/perl
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
