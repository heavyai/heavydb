namespace java com.mapd.thrift.server

enum TDatumType {
  SMALLINT,
  INT,
  BIGINT,
  FLOAT,
  DECIMAL,
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

/* union */ struct TDatumVal {
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

/* union */ struct TColumnData {
  1: list<i64> int_col,
  2: list<double> real_col,
  3: list<string> str_col,
  4: list<TColumn> arr_col
}

struct TColumn {
  1: TColumnData data,
  2: list<bool> nulls
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
  3: list<TColumn> columns
  4: bool is_columnar
}

struct TQueryResult {
  1: TRowSet row_set
  2: i64 execution_time_ms
  3: i64 total_time_ms
  4: string nonce
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

struct TCopyParams {
  1: string delimiter
  2: string null_str
  3: bool has_header
  4: bool quoted
  5: string quote
  6: string escape
  7: string line_delim
  8: string array_delim
  9: string array_begin
  10: string array_end
  11: i32 threads
}

struct TDetectResult {
  1: TRowSet row_set
  2: TCopyParams copy_params
}

struct TImportStatus {
  1: i64 elapsed
  2: i64 rows_completed
  3: i64 rows_estimated
}

struct TFrontendView {
  1: string view_name
  2: string view_state
  3: string image_hash
  4: string update_time
  5: string view_metadata
}

struct TServerStatus {
  1: bool read_only
  2: string version
  3: bool rendering_enabled
}

typedef map<string, TRenderProperty> TRenderPropertyMap
typedef map<string, TRenderPropertyMap> TColumnRenderMap

struct TPixel {
  1: i64 x
  2: i64 y
}

struct TPixelRowResult {
  1: TPixel pixel
  2: i64 row_id
  3: TRowSet row_set
  4: string nonce
}

struct TPixelRows {
  1: TPixel pixel
  2: TRowSet row_set
}

struct TPixelResult {
  1: list<TPixelRows> pixel_rows
  2: string nonce
}

struct TRenderResult {
  1: binary image
  2: string nonce
  3: i64 execution_time_ms
  4: i64 render_time_ms
  5: i64 total_time_ms
}

service MapD {
  TSessionId connect(1: string user, 2: string passwd, 3: string dbname) throws (1: TMapDException e 2: ThriftException te)
  void disconnect(1: TSessionId session) throws (1: TMapDException e 2: ThriftException te)
  TServerStatus get_server_status(1: TSessionId session) throws (1: TMapDException e 2: ThriftException te)
  TQueryResult sql_execute(1: TSessionId session, 2: string query 3: bool column_format, 4: string nonce) throws (1: TMapDException e 2: ThriftException te)
  TTableDescriptor sql_validate(1: TSessionId session, 2: string query) throws (1: TMapDException e 2: ThriftException te)
  TTableDescriptor get_table_descriptor(1: TSessionId session, 2: string table_name) throws (1: TMapDException e 2: ThriftException te)
  TRowDescriptor get_row_descriptor(1: TSessionId session, 2: string table_name) throws (1: TMapDException e 2: ThriftException te)
  TFrontendView get_frontend_view(1: TSessionId session, 2: string view_name) throws (1: TMapDException e 2: ThriftException te)
  void delete_frontend_view(1: TSessionId session, 2: string view_name) throws (1: TMapDException e 2: ThriftException te)
  list<string> get_tables(1: TSessionId session) throws (1: TMapDException e 2: ThriftException te)
  list<string> get_users() throws (1: ThriftException te)
  list<TDBInfo> get_databases() throws (1: ThriftException te)
  list<TFrontendView> get_frontend_views(1: TSessionId session) throws (1: TMapDException e 2: ThriftException te)
  void set_execution_mode(1: TSessionId session, 2: TExecuteMode mode) throws (1: TMapDException e 2: ThriftException te)
  string get_version() throws (1: ThriftException te)
  void load_table_binary(1: TSessionId session, 2: string table_name, 3: list<TRow> rows) throws (1: TMapDException e 2: ThriftException te)
  void load_table(1: TSessionId session, 2: string table_name, 3: list<TStringRow> rows) throws (1: TMapDException e 2: ThriftException te)
  TRenderResult render(1: TSessionId session, 2: string query, 3: string render_type, 4: string nonce) throws (1: TMapDException e 2: ThriftException te)
  void create_frontend_view(1: TSessionId session, 2: string view_name, 3: string view_state, 4: string image_hash, 5: string view_metadata) throws (1: TMapDException e 2: ThriftException te)
  TDetectResult detect_column_types(1: TSessionId session, 2: string file_name, 3: TCopyParams copy_params) throws (1: TMapDException e 2: ThriftException te)
  void create_table(1: TSessionId session, 2: string table_name, 3: TRowDescriptor row_desc) throws (1: TMapDException e 2: ThriftException te)
  void import_table(1: TSessionId session, 2: string table_name, 3: string file_name, 4: TCopyParams copy_params) throws (1: TMapDException e 2: ThriftException te)
  TImportStatus import_table_status(1: TSessionId session, 2: string import_id) throws (1: TMapDException e 2: ThriftException te)
  TFrontendView get_link_view(1: TSessionId session, 2: string link) throws (1: TMapDException e 2: ThriftException te)
  string create_link(1: TSessionId session, 2: string view_state, 3: string view_metadata) throws (1: TMapDException e 2: ThriftException te)
  TPixelResult get_rows_for_pixels(1: TSessionId session, 2: i64 widget_id, 3: list<TPixel> pixels, 4: string table_name, 5: list<string> col_names, 6: bool column_format, 7: string nonce) throws (1: TMapDException e 2: ThriftException te)
  TPixelRowResult get_row_for_pixel(1: TSessionId session, 2: i64 widget_id, 3: TPixel pixel, 4: string table_name, 5: list<string> col_names, 6: bool column_format, 7: i32 pixelRadius, 8: string nonce) throws (1: TMapDException e 2: ThriftException te)
  void start_heap_profile() throws (1: TMapDException e 2: ThriftException te)
  void stop_heap_profile() throws (1: TMapDException e 2: ThriftException te)
  string get_heap_profile() throws (1: TMapDException e 2: ThriftException te)
  void import_geo_table(1: TSessionId session, 2: string file_name, 3: string table_name, 4: TCopyParams copy_params) throws (1: TMapDException e 2: ThriftException te)
}
