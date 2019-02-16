include "mapd.thrift"

enum TResultSetLayout {
  GroupByPerfectHash,
  GroupByBaselineHash,
  Projection,
  NonGroupedAggregate
}

struct TSlotSize {
  1: i16 padded,
  2: i16 logical
}

struct TColSlotContext {
  1: list<TSlotSize> slot_sizes,
  2: list<list<i32>> col_to_slot_map
}

enum TCountDistinctImplType {
  Invalid,
  Bitmap,
  StdSet
}

struct TCountDistinctDescriptor {
  1: TCountDistinctImplType impl_type,
  2: i64 min_val,
  3: i64 bitmap_sz_bits,
  4: bool approximate,
  5: mapd.TDeviceType device_type,
  6: i64 sub_bitmap_count
}

typedef list<TCountDistinctDescriptor> TCountDistinctDescriptors

struct TResultSetBufferDescriptor {
  1: TResultSetLayout layout,
  2: bool keyless,
  3: i32 entry_count,
  4: i32 idx_target_as_key,
  5: i64 init_val,
  6: i64 min_val,
  7: i64 max_val,
  8: i64 bucket,
  9: list<i16> group_col_widths,
  10: i16 key_bytewidth,
  11: TColSlotContext col_slot_context,
  12: list<i32> target_groupby_indices,
  13: TCountDistinctDescriptors count_distinct_descriptors,
  14: bool force_4byte_float,
}

enum TAggKind {
  AVG, MIN, MAX, SUM, COUNT, APPROX_COUNT_DISTINCT, SAMPLE
}

struct TTargetInfo {
  1: bool is_agg,
  2: TAggKind kind,
  3: mapd.TTypeInfo type,
  4: mapd.TTypeInfo arg_type,
  5: bool skip_nulls,
  6: bool is_distinct
}

union TCountDistinctSetStorage {
  1: binary bitmap
  2: set<i64> sparse_set
}

struct TCountDistinctSet {
  1: TCountDistinctImplType type,
  2: TCountDistinctSetStorage storage,
  3: i64 remote_ptr
}

struct TSerializedRows {
  1: binary buffer,
  2: TResultSetBufferDescriptor descriptor,
  3: list<TTargetInfo> targets,
  4: list<i64> target_init_vals,
  5: list<binary> varlen_buffer,
  6: list<TCountDistinctSet> count_distinct_sets,
  7: string explanation
}
