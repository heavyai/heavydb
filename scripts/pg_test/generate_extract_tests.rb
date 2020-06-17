#!/usr/bin/env ruby

# Generate C++ statements for inclusion into gtests.
# Usage: ruby generate_extract_tests.rb
# Capture STDOUT and copy into gtests.

TIMES=[
  '1913-12-11 10:09:08',
  '1913-12-11 10:09:00',
  '1913-12-11 10:00:00',
  '1913-12-11 00:00:00',
  '1913-12-01 00:00:00',
  '1913-01-01 00:00:00',
  '1970-01-01 00:00:00',
  '2013-12-11 10:09:08',
  '2013-12-11 10:09:00',
  '2013-12-11 10:00:00',
  '2013-12-11 00:00:00',
  '2013-12-01 00:00:00',
  '2013-01-01 00:00:00']

# https://www.postgresql.org/docs/current/functions-datetime.html#FUNCTIONS-DATETIME-EXTRACT
FIELDS=[
# 'CENTURY',
  'DAY',
# 'DECADE',
  'DOW',
  'DOY',
  'EPOCH',
  'HOUR',
  'ISODOW',
# 'ISOYEAR',
  'MICROSECOND',
# 'MILLENNIUM',
  'MILLISECOND',
  'MINUTE',
  'MONTH',
  'QUARTER',
  'SECOND',
# 'TIMEZONE',
# 'TIMEZONE_HOUR',
# 'TIMEZONE_MINUTE',
  'WEEK',
  'YEAR']

QUERY="SELECT EXTRACT(%s FROM TIMESTAMP '%s');"
# Modify this command to match your PostgreSQL setup.
# (Not specifying a database defaults to the same name as the user.)
PSQL='psql -c "%s" --tuples-only'
TEST='ASSERT_EQ(%sL, v<int64_t>(run_simple_agg("%s", dt)));'

TIMES.each do |time|
  FIELDS.each do |field|
    query = QUERY % [field, time]
    psql = PSQL % query
    result = `#{psql}`
    puts TEST % [result.strip, query]
  end
end
