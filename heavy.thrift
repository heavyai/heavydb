namespace java ai.heavy.thrift.server
namespace py heavydb.thrift

include "common.thrift"
include "completion_hints.thrift"
include "QueryEngine/serialized_result_set.thrift"
include "QueryEngine/extension_functions.thrift"

enum TExecuteMode {
  GPU = 1,
  CPU
}

enum TSourceType {
  DELIMITED_FILE,
  GEO_FILE,
  PARQUET_FILE,
  RASTER_FILE,
  ODBC
}

enum TPartitionDetail {
  DEFAULT,
  REPLICATED,
  SHARDED,
  OTHER
}

enum TGeoFileLayerContents {
  EMPTY,
  GEO,
  NON_GEO,
  UNSUPPORTED_GEO
}

enum TImportHeaderRow {
  AUTODETECT,
  NO_HEADER,
  HAS_HEADER
}

enum TRole {
  SERVER, // A single node instance
  AGGREGATOR,
  LEAF,
  STRING_DICTIONARY
}

enum TTableType {
  DEFAULT,
  TEMPORARY,
  FOREIGN,
  VIEW
}

enum TTableRefreshUpdateType {
  ALL,
  APPEND
}

enum TTableRefreshTimingType {
  MANUAL,
  SCHEDULED
}

enum TTableRefreshIntervalType {
  NONE,
  HOUR,
  DAY
}

/* union */ struct TDatumVal {
  1: i64 int_val;
  2: double real_val;
  3: string str_val;
  4: list<TDatum> arr_val;
}

struct TDatum {
  1: TDatumVal val;
  2: bool is_null;
}

struct TStringValue {
  1: string str_val;
  2: bool is_null;
}

struct TColumnType {
  1: string col_name;
  2: common.TTypeInfo col_type;
  3: bool is_reserved_keyword;
  4: string src_name;
  5: bool is_system;
  6: bool is_physical;
  7: i64 col_id;
  8: optional string default_value;
}

struct TRow {
  1: list<TDatum> cols;
}

/* union */ struct TColumnData {
  1: list<i64> int_col;
  2: list<double> real_col;
  3: list<string> str_col;
  4: list<TColumn> arr_col;
}

struct TColumn {
  1: TColumnData data;
  2: list<bool> nulls;
}

struct TStringRow {
  1: list<TStringValue> cols;
}

typedef list<TColumnType> TRowDescriptor
typedef string TSessionId
typedef string TKrb5Token
typedef i64 TQueryId
typedef i64 TSubqueryId

struct TKrb5Session {
  1: TSessionId sessionId;
  2: TKrb5Token krbToken;
}

enum TMergeType {
  UNION,
  REDUCE
}

enum TRasterPointType {
  NONE,
  AUTO,
  SMALLINT,
  INT,
  FLOAT,
  DOUBLE,
  POINT
}

enum TRasterPointTransform {
  NONE,
  AUTO,
  FILE,
  WORLD
}

struct TStepResult {
  1: serialized_result_set.TSerializedRows serialized_rows;
  2: bool execution_finished;
  3: TMergeType merge_type;
  4: bool sharded;
  5: TRowDescriptor row_desc;
  6: i32 node_id;
}

struct TRowSet {
  1: TRowDescriptor row_desc;
  2: list<TRow> rows;
  3: list<TColumn> columns;
  4: bool is_columnar;
}

enum TQueryType {
  UNKNOWN,
  READ,
  WRITE,
  SCHEMA_READ,
  SCHEMA_WRITE
}

enum TArrowTransport {
  SHARED_MEMORY,
  WIRE
}

struct TQueryResult {
  1: TRowSet row_set;
  2: i64 execution_time_ms;
  3: i64 total_time_ms;
  4: string nonce;
  5: string debug;
  6: bool success=true;
  7: TQueryType query_type=TQueryType.UNKNOWN;
}

struct TDataFrame {
  1: binary sm_handle;
  2: i64 sm_size;
  3: binary df_handle;
  4: i64 df_size;
  5: i64 execution_time_ms;
  6: i64 arrow_conversion_time_ms;
  7: binary df_buffer;
}

struct TDBInfo {
  1: string db_name;
  2: string db_owner;
}

exception TDBException {
  1: string error_msg;
}

struct TCopyParams {
  1: string delimiter;
  2: string null_str;
  3: TImportHeaderRow has_header=TImportHeaderRow.AUTODETECT;
  4: bool quoted;
  5: string quote;
  6: string escape;
  7: string line_delim;
  8: string array_delim;
  9: string array_begin;
  10: string array_end;
  11: i32 threads;
  12: TSourceType source_type=TSourceType.DELIMITED_FILE;
  13: string s3_access_key;
  14: string s3_secret_key;
  15: string s3_region;
  16: common.TEncodingType geo_coords_encoding=TEncodingType.GEOINT;
  17: i32 geo_coords_comp_param=32;
  18: common.TDatumType geo_coords_type=TDatumType.GEOMETRY;
  19: i32 geo_coords_srid=4326;
  20: bool sanitize_column_names=true;
  21: string geo_layer_name;
  22: string s3_endpoint;
  23: bool geo_assign_render_groups=false;
  24: bool geo_explode_collections=false;
  25: i32 source_srid=0;
  26: string s3_session_token;
  27: TRasterPointType raster_point_type=TRasterPointType.AUTO;
  28: string raster_import_bands;
  29: i32 raster_scanlines_per_thread;
  30: TRasterPointTransform raster_point_transform=TRasterPointTransform.AUTO;
  31: bool raster_point_compute_angle=false;
  32: string raster_import_dimensions;
  33: string odbc_dsn;
  34: string odbc_connection_string;
  35: string odbc_sql_select;
  36: string odbc_sql_order_by;
  37: string odbc_username;
  38: string odbc_password;
  39: string odbc_credential_string;
  40: string add_metadata_columns;
  41: bool trim_spaces;
}

struct TCreateParams {
  1: bool is_replicated;
}

struct TDetectResult {
  1: TRowSet row_set;
  2: TCopyParams copy_params;
}

struct TImportStatus {
  1: i64 elapsed;
  2: i64 rows_completed;
  3: i64 rows_estimated;
  4: i64 rows_rejected;
}

struct TFrontendView {
  1: string view_name;
  2: string view_state;
  3: string image_hash;
  4: string update_time;
  5: string view_metadata;
}

struct TServerStatus {
  1: bool read_only;
  2: string version;
  3: bool rendering_enabled;
  4: i64 start_time;
  5: string edition;
  6: string host_name;
  7: bool poly_rendering_enabled;
  8: TRole role;
  9: string renderer_status_json;
}

struct TPixel {
  1: i64 x;
  2: i64 y;
}

struct TPixelTableRowResult {
  1: TPixel pixel;
  2: string vega_table_name;
  3: list<i64> table_id;
  4: list<i64> row_id;
  5: TRowSet row_set;
  6: string nonce;
}

struct TRenderResult {
  1: binary image;
  2: string nonce;
  3: i64 execution_time_ms;
  4: i64 render_time_ms;
  5: i64 total_time_ms;
  6: string vega_metadata;
}

struct TGpuSpecification {
  1: i32 num_sm;
  2: i64 clock_frequency_kHz;
  3: i64 memory;
  4: i16 compute_capability_major;
  5: i16 compute_capability_minor;
}

struct THardwareInfo {
  1: i16 num_gpu_hw;
  2: i16 num_cpu_hw;
  3: i16 num_gpu_allocated;
  4: i16 start_gpu;
  5: string host_name;
  6: list<TGpuSpecification> gpu_info;
}

struct TClusterHardwareInfo {
  1: list<THardwareInfo> hardware_info;
}

struct TMemoryData {
  1: i64 slab;
  2: i32 start_page;
  3: i64 num_pages;
  4: i32 touch;
  5: list<i64> chunk_key;
  6: i32 buffer_epoch;
  7: bool is_free;
}

struct TNodeMemoryInfo {
  1: string host_name;
  2: i64 page_size;
  3: i64 max_num_pages;
  4: i64 num_pages_allocated;
  5: bool is_allocation_capped;
  6: list<TMemoryData> node_memory_data;
}

struct TTableMeta {
  1: string table_name;
  2: i64 num_cols;
  4: bool is_view;
  5: bool is_replicated;
  6: i64 shard_count;
  7: i64 max_rows;
  8: i64 table_id;
  9: i64 max_table_id;
  10: list<common.TTypeInfo> col_types;
  11: list<string> col_names;
}

struct TTableRefreshInfo {
  1: TTableRefreshUpdateType update_type;
  2: TTableRefreshTimingType timing_type;
  3: string start_date_time;
  4: TTableRefreshIntervalType interval_type;
  5: i64 interval_count;
  6: string last_refresh_time;
  7: string next_refresh_time;
}

struct TTableDetails {
  1: TRowDescriptor row_desc;
  2: i64 fragment_size;
  3: i64 page_size;
  4: i64 max_rows;
  5: string view_sql;
  6: i64 shard_count;
  7: string key_metainfo;
  8: bool is_temporary;
  9: TPartitionDetail partition_detail;
  10: TTableType table_type;
  11: TTableRefreshInfo refresh_info;
}

enum TExpressionRangeType {
  INVALID,
  INTEGER,
  FLOAT,
  DOUBLE
}

struct TColumnRange {
  1: TExpressionRangeType type;
  2: i32 col_id;
  3: i32 table_id;
  4: bool has_nulls;
  5: i64 int_min;
  6: i64 int_max;
  7: i64 bucket;
  8: double fp_min;
  9: double fp_max;
}

struct TDictionaryGeneration {
  1: i32 dict_id;
  2: i64 entry_count;
}

struct TTableGeneration {
  1: i32 table_id;
  2: i64 tuple_count;
  3: i64 start_rowid;
}

struct TPendingQuery {
  1: TQueryId id;
  2: list<TColumnRange> column_ranges;
  3: list<TDictionaryGeneration> dictionary_generations;
  4: list<TTableGeneration> table_generations;
  5: TSessionId parent_session_id;
}

struct TVarLen {
  1: binary payload;
  2: bool is_null;
}

union TDataBlockPtr {
  1: binary fixed_len_data;
  2: list<TVarLen> var_len_data;
}

struct TInsertData {
  1: i32 db_id;
  2: i32 table_id;
  3: list<i32> column_ids;
  4: list<TDataBlockPtr> data;
  5: i64 num_rows;
  6: list<bool> is_default;
}

union TChunkData {
  1: binary data_buffer;
  2: binary index_buffer;
}

struct TInsertChunks {
  1: i32 db_id;
  2: i32 table_id;
  3: list<TChunkData> data;
  4: list<i64> valid_indices;
  5: i64 num_rows;
}

struct TPendingRenderQuery {
  1: TQueryId id;
}

struct TRenderParseResult {
  1: TMergeType merge_type;
  2: i32 node_id;
  3: i64 execution_time_ms;
  4: i64 render_time_ms;
  5: i64 total_time_ms;
}

struct TRawRenderPassDataResult {
  1: i32 num_pixel_channels;
  2: i32 num_pixel_samples;
  3: binary pixels;
  4: binary row_ids_A;
  5: binary row_ids_B;
  6: binary table_ids;
  7: binary accum_data;
  8: i32 accum_depth;
}

typedef map<i32, TRawRenderPassDataResult> TRenderPassMap

struct TRawPixelData {
  1: i32 width;
  2: i32 height;
  3: TRenderPassMap render_pass_map;
}

struct TRenderDatum {
  1: common.TDatumType type;
  2: i32 cnt;
  3: binary value;
}

typedef map<string, map<string, map<string, map<string, list<TRenderDatum>>>>> TRenderAggDataMap

struct TRenderStepResult {
  1: TRenderAggDataMap merge_data;
  2: TRawPixelData raw_pixel_data;
  3: i64 execution_time_ms;
  4: i64 render_time_ms;
  5: i64 total_time_ms;
}
struct TDatabasePermissions {
  1: bool create_;
  2: bool delete_;
  3: bool view_sql_editor_;
  4: bool access_;
}

struct TTablePermissions {
  1: bool create_;
  2: bool drop_;
  3: bool select_;
  4: bool insert_;
  5: bool update_;
  6: bool delete_;
  7: bool truncate_;
  8: bool alter_;
}

struct TDashboardPermissions {
  1: bool create_;
  2: bool delete_;
  3: bool view_;
  4: bool edit_;
}

struct TViewPermissions {
  1: bool create_;
  2: bool drop_;
  3: bool select_;
  4: bool insert_;
  5: bool update_;
  6: bool delete_;
}

struct TServerPermissions {
  1: bool create_;
  2: bool drop_;
  3: bool alter_;
  4: bool usage_;
}

union TDBObjectPermissions {
  1: TDatabasePermissions database_permissions_;
  2: TTablePermissions table_permissions_;
  3: TDashboardPermissions dashboard_permissions_;
  4: TViewPermissions view_permissions_;
  5: TServerPermissions server_permissions_;
}

enum TDBObjectType {
  AbstractDBObjectType = 0,
  DatabaseDBObjectType,
  TableDBObjectType,
  DashboardDBObjectType,
  ViewDBObjectType,
  ServerDBObjectType
}

struct TDBObject {
  1: string objectName;
  2: TDBObjectType objectType;
  3: list<bool> privs;
  4: string grantee;
  5: TDBObjectType privilegeObjectType;
  6: i32 objectId;
}

struct TDashboardGrantees {
  1: string name;
  2: bool is_user;
  3: TDashboardPermissions permissions;
}

struct TDashboard {
  1: string dashboard_name;
  2: string dashboard_state;
  3: string image_hash;
  4: string update_time;
  5: string dashboard_metadata;
  6: i32 dashboard_id;
  7: string dashboard_owner;
  8: bool is_dash_shared;
  9: TDashboardPermissions dashboard_permissions;
}

struct TLicenseInfo {
  1: list<string> claims;
}

struct TSessionInfo {
  1: string user;
  2: string database;
  3: i64 start_time;
  4: bool is_super;
}

struct TGeoFileLayerInfo {
  1: string name;
  2: TGeoFileLayerContents contents;
}

struct TTableEpochInfo {
  1: i32 table_id;
  2: i32 table_epoch;
  3: i32 leaf_index;
}

enum TDataSourceType {
  TABLE
}

struct TCustomExpression {
  1: i32 id;
  2: string name;
  4: string expression_json;
  5: TDataSourceType data_source_type;
  6: i32 data_source_id;
  7: bool is_deleted;
  8: string data_source_name;
}

struct TQueryInfo {
  1: string query_session_id;
  2: string query_public_session_id;
  3: string current_status;
  4: i32 executor_id;
  5: string submitted;
  6: string query_str;
  7: string login_name;
  8: string client_address;
  9: string db_name;
  10: string exec_device_type;
}

struct TLeafInfo {
  1: i32 leaf_id;
  2: i32 num_leaves;
}

service Heavy {
  # connection, admin
  TSessionId connect(1: string user, 2: string passwd, 3: string dbname) throws (1: TDBException e)
  TKrb5Session krb5_connect(1: string inputToken, 2: string dbname) throws (1: TDBException e)
  void disconnect(1: TSessionId session) throws (1: TDBException e)
  void switch_database(1: TSessionId session, 2: string dbname) throws(1: TDBException e)
  TSessionId clone_session(1: TSessionId session) throws(1: TDBException e)
  TServerStatus get_server_status(1: TSessionId session) throws (1: TDBException e)
  list<TServerStatus> get_status(1: TSessionId session) throws (1: TDBException e)
  TClusterHardwareInfo get_hardware_info(1: TSessionId session) throws (1: TDBException e)
  list<string> get_tables(1: TSessionId session) throws (1: TDBException e)
  list<string> get_tables_for_database(1: TSessionId session, 2: string database_name) throws (1: TDBException e)
  list<string> get_physical_tables(1: TSessionId session) throws (1: TDBException e)
  list<string> get_views(1: TSessionId session) throws (1: TDBException e)
  list<TTableMeta> get_tables_meta(1: TSessionId session) throws (1: TDBException e)
  TTableDetails get_table_details(1: TSessionId session, 2: string table_name) throws (1: TDBException e)
  TTableDetails get_table_details_for_database(1: TSessionId session, 2: string table_name, 3: string database_name) throws (1: TDBException e)
  TTableDetails get_internal_table_details(1: TSessionId session, 2: string table_name, 3: bool include_system_columns = true) throws (1: TDBException e)
  TTableDetails get_internal_table_details_for_database(1: TSessionId session, 2: string table_name, 3: string database_name) throws (1: TDBException e)
  list<string> get_users(1: TSessionId session) throws (1: TDBException e)
  list<TDBInfo> get_databases(1: TSessionId session) throws (1: TDBException e)
  string get_version() throws (1: TDBException e)
  void start_heap_profile(1: TSessionId session) throws (1: TDBException e)
  void stop_heap_profile(1: TSessionId session) throws (1: TDBException e)
  string get_heap_profile(1: TSessionId session) throws (1: TDBException e)
  list<TNodeMemoryInfo> get_memory(1: TSessionId session, 2: string memory_level) throws (1: TDBException e)
  void clear_cpu_memory(1: TSessionId session) throws (1: TDBException e)
  void clear_gpu_memory(1: TSessionId session) throws (1: TDBException e)
  void set_cur_session(1: TSessionId parent_session, 2: TSessionId leaf_session, 3: string start_time_str, 4: string label, 5: bool for_running_query_kernel) throws (1: TDBException e)
  void invalidate_cur_session(1: TSessionId parent_session, 2: TSessionId leaf_session, 3: string start_time_str, 4: string label, 5: bool for_running_query_kernel) throws (1: TDBException e)
  void set_table_epoch (1: TSessionId session, 2: i32 db_id, 3: i32 table_id, 4: i32 new_epoch) throws (1: TDBException e)
  void set_table_epoch_by_name (1: TSessionId session, 2: string table_name, 3: i32 new_epoch) throws (1: TDBException e)
  i32 get_table_epoch (1: TSessionId session, 2: i32 db_id, 3: i32 table_id)
  i32 get_table_epoch_by_name (1: TSessionId session, 2: string table_name)
  list<TTableEpochInfo> get_table_epochs(1: TSessionId session, 2: i32 db_id, 3: i32 table_id)
  void set_table_epochs(1: TSessionId session, 2: i32 db_id, 3: list<TTableEpochInfo> table_epochs)
  TSessionInfo get_session_info(1: TSessionId session) throws (1: TDBException e)
  list<TQueryInfo> get_queries_info(1: TSessionId session) throws (1: TDBException e)
  void set_leaf_info(1: TSessionId session, 2: TLeafInfo leaf_info) throws (1: TDBException e)
  # query, render
  TQueryResult sql_execute(1: TSessionId session, 2: string query, 3: bool column_format, 4: string nonce, 5: i32 first_n = -1, 6: i32 at_most_n = -1) throws (1: TDBException e)
  TDataFrame sql_execute_df(1: TSessionId session, 2: string query, 3: common.TDeviceType device_type, 4: i32 device_id = 0, 5: i32 first_n = -1, 6: TArrowTransport transport_method) throws (1: TDBException e)
  TDataFrame sql_execute_gdf(1: TSessionId session, 2: string query, 3: i32 device_id = 0, 4: i32 first_n = -1) throws (1: TDBException e)
  void deallocate_df(1: TSessionId session, 2: TDataFrame df, 3: common.TDeviceType device_type, 4: i32 device_id = 0) throws (1: TDBException e)
  void interrupt(1: TSessionId query_session, 2: TSessionId interrupt_session) throws (1: TDBException e)
  TRowDescriptor sql_validate(1: TSessionId session, 2: string query) throws (1: TDBException e)
  list<completion_hints.TCompletionHint> get_completion_hints(1: TSessionId session, 2: string sql, 3: i32 cursor) throws (1: TDBException e)
  void set_execution_mode(1: TSessionId session, 2: TExecuteMode mode) throws (1: TDBException e)
  TRenderResult render_vega(1: TSessionId session, 2: i64 widget_id, 3: string vega_json, 4: i32 compression_level, 5: string nonce) throws (1: TDBException e)
  TPixelTableRowResult get_result_row_for_pixel(1: TSessionId session, 2: i64 widget_id, 3: TPixel pixel, 4: map<string, list<string>> table_col_names, 5: bool column_format, 6: i32 pixelRadius, 7: string nonce) throws (1: TDBException e)

  # custom expressions
  i32 create_custom_expression(1: TSessionId session, 2: TCustomExpression custom_expression) throws (1: TDBException e)
  list<TCustomExpression> get_custom_expressions(1: TSessionId session) throws (1: TDBException e)
  void update_custom_expression(1: TSessionId session, 2: i32 id, 3: string expression_json) throws (1: TDBException e)
  void delete_custom_expressions(1: TSessionId session, 2: list<i32> custom_expression_ids, 3: bool do_soft_delete) throws (1: TDBException e)

  # dashboards
  TDashboard get_dashboard(1: TSessionId session, 2: i32 dashboard_id) throws (1: TDBException e)
  list<TDashboard> get_dashboards(1: TSessionId session) throws (1: TDBException e)
  i32 create_dashboard(1: TSessionId session, 2: string dashboard_name, 3: string dashboard_state, 4: string image_hash, 5: string dashboard_metadata) throws (1: TDBException e)
  void replace_dashboard(1: TSessionId session, 2: i32 dashboard_id, 3: string dashboard_name, 4: string dashboard_owner, 5: string dashboard_state, 6: string image_hash, 7: string dashboard_metadata) throws (1: TDBException e)
  void delete_dashboard(1: TSessionId session, 2: i32 dashboard_id) throws (1: TDBException e)
  void share_dashboards(1: TSessionId session, 2: list<i32> dashboard_ids, 3: list<string> groups, 4: TDashboardPermissions permissions) throws (1: TDBException e)
  void delete_dashboards(1: TSessionId session, 2: list<i32> dashboard_ids) throws (1: TDBException e)
  void share_dashboard(1: TSessionId session, 2: i32 dashboard_id, 3: list<string> groups, 4: list<string> objects, 5: TDashboardPermissions permissions, 6: bool grant_role = false) throws (1: TDBException e)
  void unshare_dashboard(1: TSessionId session, 2: i32 dashboard_id, 3: list<string> groups, 4: list<string> objects, 5: TDashboardPermissions permissions) throws (1: TDBException e)
  void unshare_dashboards(1: TSessionId session, 2: list<i32> dashboard_ids, 3: list<string> groups, 4: TDashboardPermissions permissions) throws (1: TDBException e)
  list<TDashboardGrantees> get_dashboard_grantees(1: TSessionId session, 2: i32 dashboard_id) throws (1: TDBException e)
  #dashboard links
  TFrontendView get_link_view(1: TSessionId session, 2: string link) throws (1: TDBException e)
  string create_link(1: TSessionId session, 2: string view_state, 3: string view_metadata) throws (1: TDBException e)
  # import
  void load_table_binary(1: TSessionId session, 2: string table_name, 3: list<TRow> rows, 4: list<string> column_names = {}) throws (1: TDBException e)
  void load_table_binary_columnar(1: TSessionId session, 2: string table_name, 3: list<TColumn> cols, 4: list<string> column_names = {}) throws (1: TDBException e)
  void load_table_binary_columnar_polys(1: TSessionId session, 2: string table_name, 3: list<TColumn> cols, 4: list<string> column_names = {}, 5: bool assign_render_groups = true) throws (1: TDBException e)
  void load_table_binary_arrow(1: TSessionId session, 2: string table_name, 3: binary arrow_stream, 4: bool use_column_names = false) throws (1: TDBException e)
  void load_table(1: TSessionId session, 2: string table_name, 3: list<TStringRow> rows, 4: list<string> column_names = {}) throws (1: TDBException e)
  TDetectResult detect_column_types(1: TSessionId session, 2: string file_name, 3: TCopyParams copy_params) throws (1: TDBException e)
  void create_table(1: TSessionId session, 2: string table_name, 3: TRowDescriptor row_desc, 4: TCreateParams create_params) throws (1: TDBException e)
  void import_table(1: TSessionId session, 2: string table_name, 3: string file_name, 4: TCopyParams copy_params) throws (1: TDBException e)
  void import_geo_table(1: TSessionId session, 2: string table_name, 3: string file_name, 4: TCopyParams copy_params, 5: TRowDescriptor row_desc, 6: TCreateParams create_params) throws (1: TDBException e)
  TImportStatus import_table_status(1: TSessionId session, 2: string import_id) throws (1: TDBException e)
  string get_first_geo_file_in_archive(1: TSessionId session, 2: string archive_path, 3: TCopyParams copy_params) throws (1: TDBException e)
  list<string> get_all_files_in_archive(1: TSessionId session, 2: string archive_path, 3: TCopyParams copy_params) throws (1: TDBException e)
  list<TGeoFileLayerInfo> get_layers_in_geo_file(1: TSessionId session, 2: string file_name, 3: TCopyParams copy_params) throws (1: TDBException e)
  # distributed
  i64 query_get_outer_fragment_count(1: TSessionId session, 2: string query) throws(1: TDBException e)
  TTableMeta check_table_consistency(1: TSessionId session, 2: i32 table_id) throws (1: TDBException e)
  TPendingQuery start_query(1: TSessionId leaf_session, 2: TSessionId parent_session, 3: string query_ra, 4: string start_time_str, 5: bool just_explain, 6: list<i64> outer_fragment_indices) throws (1: TDBException e)
  TStepResult execute_query_step(1: TPendingQuery pending_query, 2: TSubqueryId subquery_id, 3: string start_time_str) throws (1: TDBException e)
  void broadcast_serialized_rows(1: serialized_result_set.TSerializedRows serialized_rows, 2: TRowDescriptor row_desc, 3: TQueryId query_id, 4: TSubqueryId subquery_id, 5: bool is_final_subquery_result) throws (1: TDBException e)
  TPendingRenderQuery start_render_query(1: TSessionId session, 2: i64 widget_id, 3: i16 node_idx, 4: string vega_json) throws (1: TDBException e)
  TRenderStepResult execute_next_render_step(1: TPendingRenderQuery pending_render, 2: TRenderAggDataMap merged_data) throws (1: TDBException e)
  void insert_data(1: TSessionId session, 2: TInsertData insert_data) throws (1: TDBException e)
  void insert_chunks(1: TSessionId session, 2: TInsertChunks insert_chunks) throws (1: TDBException e)
  void checkpoint(1: TSessionId session, 2: i32 table_id) throws (1: TDBException e)
  # object privileges
  list<string> get_roles(1: TSessionId session) throws (1: TDBException e)
  list<TDBObject> get_db_objects_for_grantee(1: TSessionId session, 2: string roleName) throws (1: TDBException e)
  list<TDBObject> get_db_object_privs(1: TSessionId session, 2: string objectName, 3: TDBObjectType type) throws (1: TDBException e)
  list<string> get_all_roles_for_user(1: TSessionId session, 2: string userName) throws (1: TDBException e)  # NOTE: only gives direct roles, not all effective roles
  list<string> get_all_effective_roles_for_user(1: TSessionId session, 2: string userName) throws (1: TDBException e)
  bool has_role(1: TSessionId session, 2: string granteeName, 3: string roleName) throws (1: TDBException e)
  bool has_object_privilege(1: TSessionId session, 2: string granteeName, 3: string ObjectName, 4: TDBObjectType objectType, 5: TDBObjectPermissions permissions) throws (1: TDBException e)
  # licensing
  TLicenseInfo set_license_key(1: TSessionId session, 2: string key, 3: string nonce = "") throws (1: TDBException e)
  TLicenseInfo get_license_claims(1: TSessionId session, 2: string nonce = "") throws (1: TDBException e)
  # user-defined functions
  map<string, string> get_device_parameters(1: TSessionId session) throws (1: TDBException e)
  void register_runtime_extension_functions(1: TSessionId session, 2: list<extension_functions.TUserDefinedFunction> udfs, 3: list<extension_functions.TUserDefinedTableFunction> udtfs, 4: map<string, string> device_ir_map) throws (1: TDBException e)
  list<string> get_table_function_names(1: TSessionId session) throws (1: TDBException e)
  list<string> get_runtime_table_function_names(1: TSessionId session) throws (1: TDBException e)
  list<extension_functions.TUserDefinedTableFunction> get_table_function_details(1: TSessionId session, 2: list<string> udtf_names) throws (1: TDBException e)
  list<string> get_function_names(1: TSessionId session) throws (1: TDBException e)
  list<string> get_runtime_function_names(1: TSessionId session) throws (1: TDBException e)
  list<extension_functions.TUserDefinedFunction> get_function_details(1: TSessionId session, 2: list<string> udf_names) throws (1: TDBException e)
}
