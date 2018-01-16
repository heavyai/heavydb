namespace java com.mapd.thrift.server

include "completion_hints.thrift"

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
  BOOL,
  INTERVAL_DAY_TIME,
  INTERVAL_YEAR_MONTH
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

enum TDeviceType {
  CPU,
  GPU
}

enum TTableType {
  DELIMITED,
  POLYGON
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
  3: bool is_array,
  5: i32 precision,
  6: i32 scale,
  7: i32 comp_param
}

struct TColumnType {
  1: string col_name,
  2: TTypeInfo col_type,
  3: bool is_reserved_keyword,
  4: string src_name,
  5: bool is_system
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
typedef string TSessionId
typedef i64 TQueryId

enum TMergeType {
  UNION,
  REDUCE
}

struct TStepResult {
  1: string serialized_rows
  2: bool execution_finished
  3: TMergeType merge_type
  4: bool sharded
  5: TRowDescriptor row_desc
  6: i32 node_id
}

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

struct TDataFrame {
  1: binary sm_handle
  2: i64 sm_size
  3: binary df_handle
  4: i64 df_size
}

struct TDBInfo {
  1: string db_name
  2: string db_owner
}

exception TMapDException {
  1: string error_msg
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
  12: TTableType table_type=TTableType.DELIMITED
}

struct TDetectResult {
  1: TRowSet row_set
  2: TCopyParams copy_params
}

struct TImportStatus {
  1: i64 elapsed
  2: i64 rows_completed
  3: i64 rows_estimated
  4: i64 rows_rejected
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
  4: i64 start_time
  5: string edition
  6: string host_name
}

struct TPixel {
  1: i64 x
  2: i64 y
}

struct TPixelTableRowResult {
  1: TPixel pixel
  2: string vega_table_name
  3: i64 table_id
  4: i64 row_id
  5: TRowSet row_set
  6: string nonce
}

struct TRenderResult {
  1: binary image
  2: string nonce
  3: i64 execution_time_ms
  4: i64 render_time_ms
  5: i64 total_time_ms
  6: string vega_metadata
}

struct TGpuSpecification {
  1: i32 num_sm
  2: i64 clock_frequency_kHz
  3: i64 memory
  4: i16 compute_capability_major
  5: i16 compute_capability_minor
}

struct THardwareInfo {
  1: i16 num_gpu_hw
  2: i16 num_cpu_hw
  3: i16 num_gpu_allocated
  4: i16 start_gpu
  5: string host_name
  6: list<TGpuSpecification> gpu_info
}

struct TClusterHardwareInfo {
  1: list<THardwareInfo> hardware_info
}

struct TMemoryData {
  1: i64 slab
  2: i32 start_page
  3: i64 num_pages
  4: i32 touch
  5: list<i64> chunk_key
  6: i32 buffer_epoch
  7: bool is_free
}

struct TNodeMemoryInfo {
  1: string host_name
  2: i64 page_size
  3: i64 max_num_pages
  4: i64 num_pages_allocated
  5: bool is_allocation_capped
  6: list<TMemoryData> node_memory_data
}

struct TTableDetails {
  1: TRowDescriptor row_desc
  2: i64 fragment_size
  3: i64 page_size
  4: i64 max_rows
  5: string view_sql
  6: i64 shard_count
  7: string key_metainfo
  8: bool is_temporary
}

enum TExpressionRangeType {
  INVALID,
  INTEGER,
  FLOAT,
  DOUBLE
}

struct TColumnRange {
  1: TExpressionRangeType type
  2: i32 col_id
  3: i32 table_id
  4: bool has_nulls
  5: i64 int_min
  6: i64 int_max
  7: i64 bucket
  8: double fp_min
  9: double fp_max
}

struct TDictionaryGeneration {
  1: i32 dict_id
  2: i64 entry_count
}

struct TTableGeneration {
  1: i32 table_id
  2: i64 tuple_count
  3: i64 start_rowid
}

struct TPendingQuery {
  1: TQueryId id
  2: list<TColumnRange> column_ranges
  3: list<TDictionaryGeneration> dictionary_generations
  4: list<TTableGeneration> table_generations
}

struct TVarLen {
  1: binary payload
  2: bool is_null
}

union TDataBlockPtr {
  1: binary fixed_len_data
  2: list<TVarLen> var_len_data
}

struct TInsertData {
  1: i32 db_id
  2: i32 table_id
  3: list<i32> column_ids
  4: list<TDataBlockPtr> data
  5: i64 num_rows
}

struct TPendingRenderQuery {
  1: TQueryId id
}

struct TRenderParseResult {
  1: TMergeType merge_type
  2: i32 node_id
  3: i64 execution_time_ms
  4: i64 render_time_ms
  5: i64 total_time_ms
}

struct TRawRenderPassDataResult {
  1: i32 num_channels
  2: binary pixels
  3: binary row_ids_A
  4: binary row_ids_B
  5: binary table_ids
  6: binary accum_data
}

typedef map<i32, TRawRenderPassDataResult> TRenderPassMap

struct TRawPixelData {
  1: i32 width
  2: i32 height
  3: TRenderPassMap render_pass_map
}

struct TRenderDatum {
  1: TDatumType type
  2: bool is_array
  3: TDatumVal value
}

typedef map<string, map<string, map<string, map<i32, TRenderDatum>>>> TRenderDataAggMap

struct TRenderStepResult {
  1: TRenderDataAggMap merge_data
  2: TRawPixelData raw_pixel_data
  3: i64 execution_time_ms
  4: i64 render_time_ms
  5: i64 total_time_ms
}

struct TAccessPrivileges {
  1: bool select_;
  2: bool insert_;
  3: bool create_;
  4: bool truncate_;
}

enum TDBObjectType {
  AbstractDBObjectType = 0,
  DatabaseDBObjectType,
  TableDBObjectType,
  ColumnDBObjectType,
  DashboardDBObjectType
}

struct TDBObject {
  1: string objectName
  2: TDBObjectType objectType
  3: list<bool> privs
}

struct TLicenseInfo {
  1: list<string> claims
}

service MapD {
  # connection, admin
  TSessionId connect(1: string user, 2: string passwd, 3: string dbname) throws (1: TMapDException e)
  void disconnect(1: TSessionId session) throws (1: TMapDException e)
  TServerStatus get_server_status(1: TSessionId session) throws (1: TMapDException e)
  list<TServerStatus> get_status(1: TSessionId session) throws (1: TMapDException e)
  TClusterHardwareInfo get_hardware_info(1: TSessionId session) throws (1: TMapDException e)
  list<string> get_tables(1: TSessionId session) throws (1: TMapDException e)
  list<string> get_physical_tables(1: TSessionId session) throws (1: TMapDException e)
  list<string> get_views(1: TSessionId session) throws (1: TMapDException e)
  TTableDetails get_table_details(1: TSessionId session, 2: string table_name) throws (1: TMapDException e)
  TTableDetails get_internal_table_details(1: TSessionId session, 2: string table_name) throws (1: TMapDException e)
  list<string> get_users(1: TSessionId session) throws (1: TMapDException e)
  list<TDBInfo> get_databases(1: TSessionId session) throws (1: TMapDException e)
  string get_version() throws (1: TMapDException e)
  void start_heap_profile(1: TSessionId session) throws (1: TMapDException e)
  void stop_heap_profile(1: TSessionId session) throws (1: TMapDException e)
  string get_heap_profile(1: TSessionId session) throws (1: TMapDException e)
  list<TNodeMemoryInfo> get_memory(1: TSessionId session, 2: string memory_level) throws (1: TMapDException e)
  void clear_cpu_memory(1: TSessionId session) throws (1: TMapDException e)
  void clear_gpu_memory(1: TSessionId session) throws (1: TMapDException e)
  void set_table_epoch (1: TSessionId session 2: i32 db_id 3: i32 table_id 4: i32 new_epoch) throws (1: TMapDException e)
  void set_table_epoch_by_name (1: TSessionId session 2: string table_name 3: i32 new_epoch) throws (1: TMapDException e)
  i32 get_table_epoch (1: TSessionId session 2: i32 db_id 3: i32 table_id);
  i32 get_table_epoch_by_name (1: TSessionId session 2: string table_name);
  # query, render
  TQueryResult sql_execute(1: TSessionId session, 2: string query 3: bool column_format, 4: string nonce, 5: i32 first_n = -1, 6: i32 at_most_n = -1) throws (1: TMapDException e)
  TDataFrame sql_execute_df(1: TSessionId session, 2: string query 3: TDeviceType device_type 4: i32 device_id = 0 5: i32 first_n = -1) throws (1: TMapDException e)
  TDataFrame sql_execute_gdf(1: TSessionId session, 2: string query 3: i32 device_id = 0, 4: i32 first_n = -1) throws (1: TMapDException e)
  void deallocate_df(1: TSessionId session, 2: TDataFrame df, 3: TDeviceType device_type, 4: i32 device_id = 0) throws (1: TMapDException e)
  void interrupt(1: TSessionId session) throws (1: TMapDException e)
  TTableDescriptor sql_validate(1: TSessionId session, 2: string query) throws (1: TMapDException e)
  list<completion_hints.TCompletionHint> get_completion_hints(1: TSessionId session, 2:string sql, 3:i32 cursor) throws (1: TMapDException e)
  void set_execution_mode(1: TSessionId session, 2: TExecuteMode mode) throws (1: TMapDException e)
  TRenderResult render_vega(1: TSessionId session, 2: i64 widget_id, 3: string vega_json, 4: i32 compression_level, 5: string nonce) throws (1: TMapDException e)
  TPixelTableRowResult get_result_row_for_pixel(1: TSessionId session, 2: i64 widget_id, 3: TPixel pixel, 4: map<string, list<string>> table_col_names, 5: bool column_format, 6: i32 pixelRadius, 7: string nonce) throws (1: TMapDException e)
  # Immerse
  TFrontendView get_frontend_view(1: TSessionId session, 2: string view_name) throws (1: TMapDException e)
  list<TFrontendView> get_frontend_views(1: TSessionId session) throws (1: TMapDException e)
  void create_frontend_view(1: TSessionId session, 2: string view_name, 3: string view_state, 4: string image_hash, 5: string view_metadata) throws (1: TMapDException e)
  void delete_frontend_view(1: TSessionId session, 2: string view_name) throws (1: TMapDException e)
  TFrontendView get_link_view(1: TSessionId session, 2: string link) throws (1: TMapDException e)
  string create_link(1: TSessionId session, 2: string view_state, 3: string view_metadata) throws (1: TMapDException e)
  # import
  void load_table_binary(1: TSessionId session, 2: string table_name, 3: list<TRow> rows) throws (1: TMapDException e)
  void load_table_binary_columnar(1: TSessionId session, 2: string table_name, 3: list<TColumn> cols) throws (1: TMapDException e)
  void load_table_binary_arrow(1: TSessionId session, 2: string table_name, 3: binary arrow_stream) throws (1: TMapDException e)
  void load_table(1: TSessionId session, 2: string table_name, 3: list<TStringRow> rows) throws (1: TMapDException e)
  TDetectResult detect_column_types(1: TSessionId session, 2: string file_name, 3: TCopyParams copy_params) throws (1: TMapDException e)
  void create_table(1: TSessionId session, 2: string table_name, 3: TRowDescriptor row_desc, 4: TTableType table_type=TTableType.DELIMITED) throws (1: TMapDException e)
  void import_table(1: TSessionId session, 2: string table_name, 3: string file_name, 4: TCopyParams copy_params) throws (1: TMapDException e)
  void import_geo_table(1: TSessionId session, 2: string table_name, 3: string file_name, 4: TCopyParams copy_params, 5: TRowDescriptor row_desc) throws (1: TMapDException e)
  TImportStatus import_table_status(1: TSessionId session, 2: string import_id) throws (1: TMapDException e)
  # distributed
  TPendingQuery start_query(1: TSessionId session, 2: string query_ra, 3: bool just_explain) throws (1: TMapDException e)
  TStepResult execute_first_step(1: TPendingQuery pending_query) throws (1: TMapDException e)
  void broadcast_serialized_rows(1: string serialized_rows, 2: TRowDescriptor row_desc, 3: TQueryId query_id) throws (1: TMapDException e)
  TPendingRenderQuery start_render_query(1: TSessionId session, 2: i64 widget_id, 3: i16 node_idx, 4: string vega_json) throws (1: TMapDException e)
  TRenderStepResult execute_next_render_step(1: TPendingRenderQuery pending_render, 2: TRenderDataAggMap merged_data) throws (1: TMapDException e)
  void insert_data(1: TSessionId session, 2: TInsertData insert_data) throws (1: TMapDException e)
  void checkpoint(1: TSessionId session, 2: i32 db_id, 3: i32 table_id) throws (1: TMapDException e)
  # deprecated
  TTableDescriptor get_table_descriptor(1: TSessionId session, 2: string table_name) throws (1: TMapDException e)
  TRowDescriptor get_row_descriptor(1: TSessionId session, 2: string table_name) throws (1: TMapDException e)
  # object privileges
  list<string> get_role(1: TSessionId session 2: string roleName  3: bool userPrivateRole) throws (1: TMapDException e)
  list<string> get_all_roles(1: TSessionId session 2: bool userPrivateRole) throws (1: TMapDException e)
  list<TAccessPrivileges> get_db_object_privileges_for_role(1: TSessionId session 2: string roleName 3: i16 objectType 4: string objectName) throws (1: TMapDException e)
  list<TDBObject> get_db_objects_for_role(1: TSessionId session 2: string roleName) throws (1: TMapDException e)
  list<TDBObject> get_db_object_privs(1: TSessionId session 2: string objectName) throws (1: TMapDException e)
  list<string> get_all_roles_for_user(1: TSessionId session 2: string userName) throws (1: TMapDException e)
  list<TAccessPrivileges> get_db_object_privileges_for_user(1: TSessionId session 2: string userName 3: i16 objectType 4: string objectName) throws (1: TMapDException e)
  list<TDBObject> get_db_objects_for_user(1: TSessionId session 2: string userName) throws (1: TMapDException e)
  # licensing
  TLicenseInfo set_license_key(1: TSessionId session, 2: string key, 3: string nonce = "") throws (1: TMapDException e)
  TLicenseInfo get_license_claims(1: TSessionId session, 2: string nonce = "") throws (1: TMapDException e)
}
