enum TDatumType {
  INT,
  REAL,
  STR
}

union TDatum {
  1: TDatumType type,
  2: i64 int_val,
  3: double real_val,
  4: string str_val
}

typedef list<TDatum> TResultRow
typedef list<TResultRow> TResultRowSet
typedef list<string> TResultProjNames

struct QueryResult {
  1: TResultProjNames proj_names
  2: TResultRowSet rows
}

exception InvalidQueryException {
  1: string error_msg
}

service MapD {
  QueryResult select(1: string query) throws (1: InvalidQueryException e)
}
