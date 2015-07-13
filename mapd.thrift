enum TDatumType {
  INT,
  FLOAT,
  DOUBLE,
  STR,
  TIME,
  TIMESTAMP,
  DATE,
  BOOL
}

enum TEncodingType {
  NONE,
  FIXED,
  RL,
  DIFF,
  DICT,
  SPARSE
}

enum TExecuteMode {
  HYBRID,
  GPU,
  CPU
}

union TDatumVal {
  1: i64 int_val,
  2: double real_val,
  3: string str_val,
  4: list<TDatum> arr_val
}

struct TDatum {
  1: TDatumVal val,
  2: bool is_null
}

struct TStringValue {
  1: string str_val
  2: bool is_null
}

struct TTypeInfo {
  1: TDatumType type,
  4: TEncodingType encoding,
  2: bool nullable,
  3: bool is_array
}

struct TColumnType {
  1: string col_name,
  2: TTypeInfo col_type
}

struct TRow {
  1: list<TDatum> cols
}

struct TStringRow {
  1: list<TStringValue> cols
}

typedef list<TColumnType> TRowDescriptor
typedef map<string, TColumnType> TTableDescriptor
typedef i32 TSessionId

struct TRowSet {
  1: TRowDescriptor row_desc
  2: list<TRow> rows
}

struct TQueryResult {
  1: TRowSet row_set
  2: i64 execution_time_ms
}

struct TDBInfo {
  1: string db_name
  2: string db_owner
}

exception TMapDException {
  1: string error_msg
}

exception ThriftException {
  1: string error_msg
}

struct TRenderProperty {
  1: TDatumType property_type
  2: TDatumVal property_value
}

typedef map<string, TRenderProperty> TRenderPropertyMap
typedef map<string, TRenderPropertyMap> TColumnRenderMap

service MapD {
  TSessionId connect(1: string user, 2: string passwd, 3: string dbname) throws (1: TMapDException e 2: ThriftException te)
  void disconnect(1: TSessionId session) throws (1: TMapDException e 2: ThriftException te)
  TQueryResult sql_execute(1: TSessionId session, 2: string query) throws (1: TMapDException e 2: ThriftException te)
  TTableDescriptor get_table_descriptor(1: TSessionId session, 2: string table_name) throws (1: TMapDException e 2: ThriftException te)
  TRowDescriptor get_row_descriptor(1: TSessionId session, 2: string table_name) throws (1: TMapDException e 2: ThriftException te)
  string get_frontend_view(1: TSessionId session,  2: string view_name) throws (1: TMapDException e 2: ThriftException te)
  list<string> get_tables(1: TSessionId session) throws (1: TMapDException e 2: ThriftException te)
  list<string> get_users() throws (1: ThriftException te)
  list<TDBInfo> get_databases() throws (1: ThriftException te)
  list<string> get_frontend_views(1: TSessionId session) throws (1: TMapDException e 2: ThriftException te)
  void set_execution_mode(1: TSessionId session, 2: TExecuteMode mode) throws (1: TMapDException e 2: ThriftException te)
  string get_version() throws (1: ThriftException te)
  void load_table_binary(1: TSessionId session, 2: string table_name, 3: list<TRow> rows) throws (1: TMapDException e 2: ThriftException te)
  void load_table(1: TSessionId session, 2: string table_name, 3: list<TStringRow> rows) throws (1: TMapDException e 2: ThriftException te)
  binary render(1: TSessionId session, 2: string query, 3: string render_type, 4: TRenderPropertyMap render_properties, 5: TColumnRenderMap col_render_properties) throws (1: TMapDException e 2: ThriftException te)
  void create_frontend_view(1: TSessionId session, 2: string view_name, 3: string view_state) throws (1: TMapDException e 2: ThriftException te)
}
