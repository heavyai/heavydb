#!/usr/bin/ruby
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Show minimum execution times from generated report*.txt files.

# Usage:
# $ grep Execution result*.txt | ./report.rb

report = {} # query num -> Array of times

ARGF.each do |line|
  md = line.match(/result(\d+).txt:Execution time: (\d+) ms/) or raise "Unknown format: #{line}"
  report[md[1].to_i] ||= []
  report[md[1].to_i] << md[2].to_i
end

report.keys.sort.each do |key|
  puts "#{key},#{report[key].min}"
end
