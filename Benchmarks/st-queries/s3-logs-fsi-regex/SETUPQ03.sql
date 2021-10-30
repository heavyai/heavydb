CREATE FOREIGN TABLE ##TAB##  (
  log_timestamp TIMESTAMP(0),
  severity TEXT ENCODING DICT(32),
  process_id INTEGER,
  file_name TEXT ENCODING DICT(32),
  api_name TEXT ENCODING DICT(32),
  duration_ms BIGINT,
  db_name TEXT ENCODING DICT(32),
  user_name TEXT ENCODING DICT(32),
  public_session_id TEXT ENCODING DICT(32),
  field_names TEXT[] ENCODING DICT(32),
  field_values TEXT[] ENCODING DICT(32))
SERVER benchmark_s3_regex_parser
WITH (FILE_PATH = '##FILE##',
      LINE_REGEX = '^([^\s]+)\s(\w)\s(\d+)\s([^\s]+)\s(?:stdlog)\s(\w+)\s(?:\d+)\s(\d+)\s(\w+)\s([^\s]+)\s([^\s]+)\s(\{[^\}]+\})\s(\{[^\}]+\})$')
