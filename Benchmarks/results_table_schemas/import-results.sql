CREATE TABLE ##TAB##
  (
    run_guid TEXT ENCODING DICT(32),
    run_timestamp TIMESTAMP(0),
    run_connection TEXT ENCODING DICT(32),
    run_machine_name TEXT ENCODING DICT(32),
    run_machine_uname TEXT ENCODING DICT(32),
    run_driver TEXT ENCODING DICT(32),
    run_version TEXT ENCODING DICT(32),
    run_label TEXT ENCODING DICT(32),
    import_elapsed_time_ms DOUBLE,
    import_execute_time_ms DOUBLE,
    import_conn_time_ms DOUBLE,
    rows_loaded TEXT ENCODING DICT(32),
    rows_rejected TEXT ENCODING DICT(32)
  )
