enum DatumType {
  INT,
  REAL,
  STR
}

union Datum {
  1: DatumType type,
  2: i64 int_val,
  3: double real_val,
  4: string str_val
}

typedef list<Datum> ResultRow
typedef list<ResultRow> ResultRowSet
typedef list<string> ResultProjNames

struct QueryResult {
  1: ResultProjNames proj_names
  2: ResultRowSet rows
}

service MapD {
  QueryResult select(1: string query)
}
