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


service MapD {
  ResultRowSet select(1: string query)
}
