namespace php mapd

enum TDatumType {
  INT,
  REAL,
  STR,
  TIME
}

union TDatum {
  1: i64 int_val,
  2: double real_val,
  3: string str_val
}

struct ColumnValue {
  1: TDatumType type,
  2: TDatum datum
}

struct ColumnType {
  1: TDatumType type,
  2: bool nullable
}

typedef list<ColumnValue> TResultRow
typedef list<TResultRow> TResultRowSet
typedef list<string> TResultProjNames
typedef map<string, ColumnType> ColumnTypes

struct QueryResult {
  1: TResultProjNames proj_names
  2: TResultRowSet rows
}

exception MapDException {
  1: string error_msg
}

service MapD {
  QueryResult select(1: string query) throws (1: MapDException e)
  ColumnTypes getColumnTypes(1: string table_name) throws (1: MapDException e)
}
