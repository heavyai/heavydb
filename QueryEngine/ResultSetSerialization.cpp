#include "../MapDServer.h"
#include "Execute.h"
#include "GroupByAndAggregate.h"
#include "ResultSet.h"
#include "Shared/likely.h"
#include "Shared/mapd_shared_ptr.h"
#include "Shared/scope.h"
#include "ThriftSerializers.h"
#include "gen-cpp/serialized_result_set_types.h"

#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/transport/TBufferTransports.h>
#include <boost/smart_ptr/make_shared.hpp>

#include <future>

namespace {

bool has_varlen_agg_targets(const std::vector<TargetInfo>& targets) {
  bool ret = false;
  for (const auto& target : targets) {
    if (target.is_agg && (target.sql_type.is_varlen() ||
                          (target.sql_type.is_string() &&
                           target.sql_type.get_compression() == kENCODING_NONE))) {
      ret = true;
    }
  }
  return ret;
}

}  // namespace

std::string ResultSet::serialize() const {
  if (!just_explain_ &&
      (query_mem_desc_.getQueryDescriptionType() == QueryDescriptionType::Projection ||
       !permutation_.empty())) {
    return serializeProjection();
  }
  auto buffer = mapd::make_shared<apache::thrift::transport::TMemoryBuffer>();
  auto proto = mapd::make_shared<apache::thrift::protocol::TBinaryProtocol>(buffer);
  TSerializedRows serialized_rows;
  if (!just_explain_) {
    if (storage_) {
      const auto storage_buffer =
          reinterpret_cast<const char*>(storage_->getUnderlyingBuffer());
      serialized_rows.buffer = std::string(
          storage_buffer, storage_->query_mem_desc_.getBufferSizeBytes(device_type_));
      if (has_varlen_agg_targets(targets_)) {
        std::vector<std::string> varlen_buffer;
        serializeVarlenAggColumn(
            reinterpret_cast<int8_t*>(const_cast<char*>(serialized_rows.buffer.data())),
            varlen_buffer);
        serialized_rows.varlen_buffer.swap(varlen_buffer);
      }
      serialized_rows.target_init_vals = storage_->target_init_vals_;
      serializeCountDistinctColumns(serialized_rows);
    }
    serialized_rows.descriptor = QueryMemoryDescriptor::toThrift(query_mem_desc_);
    serialized_rows.targets = ThriftSerializers::target_infos_to_thrift(targets_);
  } else {
    serialized_rows.explanation = explanation_;
  }
  serialized_rows.write(proto.get());
  return buffer->getBufferAsString();
}

namespace {

template <typename T>
struct VarlenScalarTargetValueSize {
  static auto get_transfer_size(int) { return sizeof(T); }
};

template <>
struct VarlenScalarTargetValueSize<int64_t> {
  static auto get_transfer_size(int runtime_size_hint) {
    auto transfer_size = static_cast<unsigned int>(runtime_size_hint);
    if (UNLIKELY(transfer_size > 8 || __builtin_popcount(transfer_size) > 1)) {
      LOG(FATAL) << "Cannot serialize type. Unrecognized type size: " << transfer_size;
    }
    return transfer_size;
  }
};

struct VarlenScalarTargetValueVisitor : public boost::static_visitor<void> {
  VarlenScalarTargetValueVisitor(const int element_size) : element_size(element_size) {}

  template <typename T>
  void operator()(const T value) {
    static_assert(
        std::is_fundamental<T>::value,
        "Unsupported operator type for varlen serializer scalar target visitor");

    auto transfer_size = VarlenScalarTargetValueSize<T>::get_transfer_size(element_size);
    auto* value_ptr = reinterpret_cast<const char*>(&value);
    std::copy(value_ptr, value_ptr + transfer_size, std::back_inserter(array_buf));
  }

  void operator()(const NullableString& s) {
    const auto str_ptr = boost::get<std::string>(&s);
    if (str_ptr) {
      array_buf.insert(array_buf.end(), *str_ptr->c_str(), str_ptr->length());
    }
  }

  std::vector<char> array_buf;
  int element_size;
};

struct GeoTargetValueVistor : public boost::static_visitor<void> {
  void operator()(const GeoPointTargetValue& point_tv) {
    auto coords_ptr = reinterpret_cast<const char*>(point_tv.coords->data());
    varlen_buf.emplace_back(coords_ptr,
                            coords_ptr + point_tv.coords->size() * sizeof(double));
  }
  void operator()(const GeoLineStringTargetValue& linestring_tv) {
    auto coords_ptr = reinterpret_cast<const char*>(linestring_tv.coords->data());
    varlen_buf.emplace_back(coords_ptr,
                            coords_ptr + linestring_tv.coords->size() * sizeof(double));
  }
  void operator()(const GeoPolyTargetValue& poly_tv) {
    auto coords_ptr = reinterpret_cast<const char*>(poly_tv.coords->data());
    varlen_buf.emplace_back(coords_ptr,
                            coords_ptr + poly_tv.coords->size() * sizeof(double));

    auto ring_sz_ptr = reinterpret_cast<const char*>(poly_tv.ring_sizes->data());
    varlen_buf.emplace_back(ring_sz_ptr,
                            ring_sz_ptr + poly_tv.ring_sizes->size() * sizeof(int32_t));
  }
  void operator()(const GeoMultiPolyTargetValue& mpoly_tv) {
    auto coords_ptr = reinterpret_cast<const char*>(mpoly_tv.coords->data());
    varlen_buf.emplace_back(coords_ptr,
                            coords_ptr + mpoly_tv.coords->size() * sizeof(double));

    auto ring_sz_ptr = reinterpret_cast<const char*>(mpoly_tv.ring_sizes->data());
    varlen_buf.emplace_back(ring_sz_ptr,
                            ring_sz_ptr + mpoly_tv.ring_sizes->size() * sizeof(int32_t));

    auto poly_rings_ptr = reinterpret_cast<const char*>(mpoly_tv.poly_rings->data());
    varlen_buf.emplace_back(
        poly_rings_ptr, poly_rings_ptr + mpoly_tv.poly_rings->size() * sizeof(int32_t));
  }

  std::vector<std::string> varlen_buf;
};

void serialize_projected_column(int8_t* col_ptr,
                                std::vector<std::string>& varlen_buffer,
                                const TargetValue& tv,
                                const SQLTypeInfo& ti) {
  if (ti.is_string() && ti.get_compression() == kENCODING_NONE) {
    const auto scalar_tv = boost::get<ScalarTargetValue>(&tv);
    CHECK(scalar_tv);
    const auto nullable_str = boost::get<NullableString>(scalar_tv);
    const auto str_p = boost::get<std::string>(nullable_str);
    if (str_p) {
      *reinterpret_cast<int32_t*>(col_ptr) = varlen_buffer.size();
      varlen_buffer.push_back(*str_p);
    } else {
      *reinterpret_cast<int32_t*>(col_ptr) = -1;
    }
    return;
  }
  if (ti.is_array()) {
    const auto arr_tv = boost::get<ArrayTargetValue>(&tv);
    CHECK(arr_tv);
    if (arr_tv->is_initialized()) {
      const auto& vec = arr_tv->get();
      *reinterpret_cast<int32_t*>(col_ptr) = varlen_buffer.size();
      if (ti.is_string_array()) {
        CHECK(ti.get_compression() == kENCODING_DICT);
      }

      const auto elem_ti = ti.get_elem_type();
      const auto elem_sz = ti.is_string_array() ? 4 : elem_ti.get_size();
      VarlenScalarTargetValueVisitor varlen_visitor(elem_sz);
      for (const auto& e : vec) {
        boost::apply_visitor(varlen_visitor, e);
      }
      varlen_buffer.emplace_back(varlen_visitor.array_buf.begin(),
                                 varlen_visitor.array_buf.end());
    }
    return;
  }
  if (ti.is_geometry()) {
    const auto geo_target_value = boost::get<GeoTargetValue>(&tv);
    CHECK(geo_target_value);
    *reinterpret_cast<int32_t*>(col_ptr) = varlen_buffer.size();
    GeoTargetValueVistor geo_visitor;
    boost::apply_visitor(geo_visitor, *geo_target_value);
    varlen_buffer.insert(varlen_buffer.end(),
                         geo_visitor.varlen_buf.begin(),
                         geo_visitor.varlen_buf.end());
    return;
  }
  int64_t int_val{0};
  if (ti.is_integer() || ti.is_decimal() || ti.is_boolean() || ti.is_time() ||
      ti.is_timeinterval() || ti.is_string()) {
    const auto scalar_tv = boost::get<ScalarTargetValue>(&tv);
    CHECK(scalar_tv);
    const auto i64_p = boost::get<int64_t>(scalar_tv);
    CHECK(i64_p);
    int_val = *i64_p;
  }
  double double_val{0};
  float float_val{0};
  if (ti.is_fp()) {
    const auto scalar_tv = boost::get<ScalarTargetValue>(&tv);
    CHECK(scalar_tv);
    if (ti.get_type() == kDOUBLE) {
      const auto double_p = boost::get<double>(scalar_tv);
      CHECK(double_p);
      double_val = *double_p;
    } else {
      CHECK_EQ(kFLOAT, ti.get_type());
      const auto float_p = boost::get<float>(scalar_tv);
      CHECK(float_p);
      float_val = *float_p;
    }
  }
  switch (ti.get_type()) {
    case kBOOLEAN:
    case kTINYINT: {
      *reinterpret_cast<int8_t*>(col_ptr) = int_val;
      break;
    }
    case kSMALLINT: {
      *reinterpret_cast<int16_t*>(col_ptr) = int_val;
      break;
    }
    case kINT:
    case kCHAR:
    case kVARCHAR:
    case kTEXT: {
      *reinterpret_cast<int32_t*>(col_ptr) = int_val;
      break;
    }
    case kBIGINT:
    case kNUMERIC:
    case kDECIMAL:
    case kDATE:
    case kTIMESTAMP:
    case kTIME:
    case kINTERVAL_DAY_TIME:
    case kINTERVAL_YEAR_MONTH: {
      *reinterpret_cast<int64_t*>(col_ptr) = int_val;
      break;
    }
    case kFLOAT: {
      *reinterpret_cast<float*>(col_ptr) = float_val;
      break;
    }
    case kDOUBLE: {
      *reinterpret_cast<double*>(col_ptr) = double_val;
      break;
    }
    default:
      throw std::runtime_error("Unable to serialize type `" + ti.get_type_name() +
                               "` for distributed projection.");
  }
}

size_t get_projected_size(const SQLTypeInfo& ti) {
  return (ti.is_array() || ti.is_geometry() ||
          (ti.is_string() && ti.get_compression() == kENCODING_NONE))
             ? 4
             : ti.get_logical_size();
}

}  // namespace

// The projection layout could contain lazy values, real strings and arrays.
// Leverage the high-level row iteration to retrieve the values and put them
// in the address-space independent buffer which then goes over the network.
std::string ResultSet::serializeProjection() const {
  moveToBegin();
  ScopeGuard restore_cursor = [this] { moveToBegin(); };
  size_t one_row_size{8};  // Store the index of the row for now, although it's
                           // redundant since we only serialize non-empty entries.

  auto proj_query_mem_desc = query_mem_desc_;
  proj_query_mem_desc.setQueryDescriptionType(QueryDescriptionType::Projection);
  proj_query_mem_desc.setHasKeylessHash(false);
  proj_query_mem_desc.resetGroupColWidths({8});
  proj_query_mem_desc.clearTargetGroupbyIndices();
  proj_query_mem_desc.clearSlotInfo();

  // Note: When projecting columns, floats are serialized in 4 bytes as opposed to 8 bytes
  // coming from the executor. For now we add an override.
  proj_query_mem_desc.setForceFourByteFloat(true);

  // When projecting geo, ensure we return GeoTargetValue and not a string
  const auto cur_geo_return_type = getGeoReturnType();
  ScopeGuard restore_geo_return_type = [this, &cur_geo_return_type] {
    geo_return_type_ = cur_geo_return_type;
  };
  geo_return_type_ = GeoReturnType::GeoTargetValue;

  for (size_t i = 0; i < colCount(); ++i) {
    const auto ti = getColType(i);
    const int8_t logical_size = get_projected_size(ti);
    proj_query_mem_desc.addColSlotInfo({std::make_tuple(logical_size, logical_size)});
    one_row_size += logical_size;
  }
  std::unique_ptr<int8_t[]> serialized_storage(new int8_t[one_row_size * entryCount()]);
  auto row_ptr = serialized_storage.get();
  size_t row_count{0};
  std::vector<std::string> varlen_buffer;
  while (true) {
    const auto crt_row = getNextRow(false, false);
    if (crt_row.empty()) {
      break;
    }
    CHECK_EQ(colCount(), crt_row.size());
    *reinterpret_cast<int64_t*>(row_ptr) = row_count;
    auto col_ptr = row_ptr + 8;
    for (size_t i = 0; i < colCount(); ++i) {
      const auto ti = getColType(i);
      serialize_projected_column(col_ptr, varlen_buffer, crt_row[i], ti);
      col_ptr += get_projected_size(ti);
    }
    ++row_count;
    row_ptr += one_row_size;
  }
  proj_query_mem_desc.setEntryCount(row_count);
  TSerializedRows serialized_rows;
  serialized_rows.buffer = std::string(
      reinterpret_cast<const char*>(serialized_storage.get()), one_row_size * row_count);
  serialized_rows.descriptor = QueryMemoryDescriptor::toThrift(proj_query_mem_desc);
  serialized_rows.targets = ThriftSerializers::target_infos_to_thrift(targets_);
  serialized_rows.varlen_buffer.swap(varlen_buffer);
  auto buffer = mapd::make_shared<apache::thrift::transport::TMemoryBuffer>();
  auto proto = mapd::make_shared<apache::thrift::protocol::TBinaryProtocol>(buffer);
  serialized_rows.write(proto.get());
  return buffer->getBufferAsString();
}

static constexpr size_t TEST_BLOCK_SIZE = 2000;
static const char testblock[TEST_BLOCK_SIZE]{};

inline bool is_block_all_zeros(const int8_t* block, size_t block_size) {
  size_t crt_offset = 0;

  while (static_cast<int>(crt_offset * TEST_BLOCK_SIZE) <
         static_cast<int>(block_size - TEST_BLOCK_SIZE)) {
    if (memcmp(testblock, block + crt_offset * TEST_BLOCK_SIZE, TEST_BLOCK_SIZE)) {
      return false;
    }
    ++crt_offset;
  }

  for (size_t i = crt_offset * TEST_BLOCK_SIZE; i < block_size; i++) {
    if (block[i] != 0) {
      return false;
    }
  }
  return true;
}

void ResultSet::serializeVarlenAggColumn(int8_t* buf,
                                         std::vector<std::string>& varlen_buffer) const {
  auto entry_count = query_mem_desc_.getEntryCount();
  CHECK_GT(entry_count, size_t(0));

  CHECK(!query_mem_desc_.didOutputColumnar());

  // When projecting geo, ensure we return GeoTargetValue and not a string
  const auto cur_geo_return_type = getGeoReturnType();
  ScopeGuard restore_geo_return_type = [this, &cur_geo_return_type] {
    geo_return_type_ = cur_geo_return_type;
  };
  geo_return_type_ = GeoReturnType::GeoTargetValue;

  if (query_mem_desc_.getQueryDescriptionType() ==
      QueryDescriptionType::NonGroupedAggregate) {
    throw std::runtime_error(
        "Projection of variable length aggregates without a group by is not yet "
        "supported in Distributed mode");
  } else {
    for (size_t i = 0; i < entry_count; ++i) {
      if (storage_->isEmptyEntry(i)) {
        continue;
      }

      const auto key_bytes = get_key_bytes_rowwise(query_mem_desc_);
      const auto key_bytes_with_padding = align_to_int64(key_bytes);
      auto target_ptr = row_ptr_rowwise(buf, query_mem_desc_, i) + key_bytes_with_padding;

      size_t target_slot_idx = 0;
      for (size_t target_logical_idx = 0; target_logical_idx < targets_.size();
           ++target_logical_idx) {
        const auto& target_info = targets_[target_logical_idx];
        if (target_info.is_agg && target_info.sql_type.is_varlen()) {
          auto compact_sz1 = query_mem_desc_.getPaddedColumnWidthBytes(target_slot_idx);

          int8_t* ptr2{nullptr};
          int8_t compact_sz2{0};
          if (!target_info.sql_type.is_geometry()) {
            ptr2 = const_cast<int8_t*>(target_ptr) + compact_sz1;
            compact_sz2 = query_mem_desc_.getPaddedColumnWidthBytes(target_slot_idx + 1);
          }

          int32_t col_ptr = -1;
          serialize_projected_column(
              reinterpret_cast<int8_t*>(&col_ptr),
              varlen_buffer,
              target_info.sql_type.is_geometry()
                  ? makeGeoTargetValue(
                        target_ptr, target_slot_idx, target_info, target_logical_idx, i)
                  : makeVarlenTargetValue(target_ptr,
                                          compact_sz1,
                                          ptr2,
                                          compact_sz2,
                                          target_info,
                                          target_logical_idx,
                                          false,
                                          i),
              target_info.sql_type);
          // Replace the entire 8-byte slot with the ptr offset
          switch (compact_sz1) {
            case 4:
              *(reinterpret_cast<int32_t*>(target_ptr)) = static_cast<int32_t>(col_ptr);
              break;
            case 8:
              *(reinterpret_cast<int64_t*>(target_ptr)) = static_cast<int64_t>(col_ptr);
              break;
            default:
              UNREACHABLE()
                  << "Invalid size for varlen pointer in aggregate buffer serialization.";
          }
        }

        target_ptr = advance_target_ptr_row_wise(
            target_ptr, target_info, target_slot_idx, query_mem_desc_, false);
        target_slot_idx = advance_slot(target_slot_idx, target_info, false);
      }
    }
  }
}

void ResultSet::serializeCountDistinctColumns(TSerializedRows& serialized_rows) const {
  BufferSet count_distinct_active_buffer_set;
  // If the count distinct query ran on GPU, all bitmaps come from a single,
  // contiguous buffer which needs to be skipped since the beginning of the first
  // logical bitmap buffer is the same as this contiguous buffer.
  const auto bitmap_pool_buffers = std::count_if(
      row_set_mem_owner_->count_distinct_bitmaps_.begin(),
      row_set_mem_owner_->count_distinct_bitmaps_.end(),
      [](const RowSetMemoryOwner::CountDistinctBitmapBuffer& count_distinct_buffer) {
        return !count_distinct_buffer.system_allocated;
      });

  // find and track active CountDistinctBuffers still in the current resultset
  create_active_buffer_set(count_distinct_active_buffer_set);

  size_t added_bitmaps = 0;
  size_t reviewed_bitmaps = 0;
  size_t zero_block_count = 0;

  for (const auto& bitmap : row_set_mem_owner_->count_distinct_bitmaps_) {
    reviewed_bitmaps++;
    if (bitmap_pool_buffers && bitmap.system_allocated) {
      continue;
    }
    // check bit map is in current active set
    const auto it = count_distinct_active_buffer_set.find(
        reinterpret_cast<const int64_t>(bitmap.ptr));
    if (it == count_distinct_active_buffer_set.end()) {
      // not an active buffer
      continue;
    }

    // check if bitmap is all zero, don't transport zero buffers rebuild on other side
    if (is_block_all_zeros(bitmap.ptr, bitmap.size)) {
      zero_block_count++;
      continue;
    }
    added_bitmaps++;

    TCountDistinctSet thrift_bitmap;
    thrift_bitmap.type = TCountDistinctImplType::Bitmap;
    thrift_bitmap.storage.__set_bitmap(
        std::string(reinterpret_cast<const char*>(bitmap.ptr), bitmap.size));
    thrift_bitmap.remote_ptr = reinterpret_cast<const int64_t>(bitmap.ptr);
    serialized_rows.count_distinct_sets.emplace_back(thrift_bitmap);
  }

  if (reviewed_bitmaps) {
    LOG(INFO) << "Serialization loop processed " << reviewed_bitmaps << " items"
              << ". Total to be serialized " << added_bitmaps
              << ". Number of zero blocks " << zero_block_count;
  }
  for (const auto sparse_set : row_set_mem_owner_->count_distinct_sets_) {
    TCountDistinctSet thrift_sparse_set;
    thrift_sparse_set.type = TCountDistinctImplType::StdSet;
    thrift_sparse_set.storage.__set_sparse_set(*sparse_set);
    thrift_sparse_set.remote_ptr = reinterpret_cast<const int64_t>(sparse_set);
    serialized_rows.count_distinct_sets.emplace_back(thrift_sparse_set);
  }
}

void ResultSet::create_active_buffer_set(
    BufferSet& count_distinct_active_buffer_set) const {
  if (query_mem_desc_.didOutputColumnar()) {
    LOG(FATAL) << "Trying to serialize leaf results sets, columnar layout not supported "
                  "currently";
  }
  for (size_t i = 0; i < query_mem_desc_.getEntryCount(); ++i) {
    const auto storage_lookup_result = StorageLookupResult{storage_.get(), i, 0};
    const auto storage = storage_lookup_result.storage_ptr;
    const auto local_entry_idx = storage_lookup_result.fixedup_entry_idx;

    const auto buff = storage->buff_;
    CHECK(buff);

    size_t agg_col_idx = 0;
    int8_t* rowwise_target_ptr{nullptr};
    int8_t* keys_ptr{nullptr};

    keys_ptr = row_ptr_rowwise(buff, query_mem_desc_, local_entry_idx);
    const auto key_bytes_with_padding =
        align_to_int64(get_key_bytes_rowwise(query_mem_desc_));
    rowwise_target_ptr = keys_ptr + key_bytes_with_padding;
    for (size_t target_idx = 0; target_idx < storage_->targets_.size(); ++target_idx) {
      const auto& agg_info = storage_->targets_[target_idx];
      auto ref_ptr = getDistinctBufferRefFromBufferRowwise(rowwise_target_ptr, agg_info);
      if (ref_ptr != int64_t{-1}) {
        count_distinct_active_buffer_set.emplace(ref_ptr);
      }
      rowwise_target_ptr = advance_target_ptr_row_wise(rowwise_target_ptr,
                                                       agg_info,
                                                       agg_col_idx,
                                                       query_mem_desc_,
                                                       separate_varlen_storage_valid_);

      agg_col_idx = advance_slot(agg_col_idx, agg_info, separate_varlen_storage_valid_);
    }
  }
}

// return the count distinct ref ptr for rowise  ( return -1 if not a count distinct
// field)
int64_t ResultSet::getDistinctBufferRefFromBufferRowwise(
    int8_t* rowwise_target_ptr,
    const TargetInfo& target_info) const {
  if (is_distinct_target(target_info)) {
    auto count_distinct_ptr_ptr = reinterpret_cast<int64_t*>(rowwise_target_ptr);
    const auto remote_ptr = *count_distinct_ptr_ptr;
    if (remote_ptr) {
      return remote_ptr;
    }
  }
  return int64_t{-1};
}

std::unique_ptr<ResultSet> ResultSet::unserialize(const std::string& str,
                                                  const Executor* executor) {
  auto buffer_bytes = reinterpret_cast<uint8_t*>(const_cast<char*>(str.data()));
  auto buffer = mapd::make_shared<apache::thrift::transport::TMemoryBuffer>(buffer_bytes,
                                                                            str.size());
  auto proto = mapd::make_shared<apache::thrift::protocol::TBinaryProtocol>(buffer);
  TSerializedRows serialized_rows;
  serialized_rows.read(proto.get());
  if (!serialized_rows.explanation.empty()) {
    return std::make_unique<ResultSet>(serialized_rows.explanation);
  }
  const auto target_infos =
      ThriftSerializers::target_infos_from_thrift(serialized_rows.targets);
  auto query_mem_desc = QueryMemoryDescriptor(serialized_rows.descriptor);
  CHECK(executor);
  auto row_set_mem_owner = executor->getRowSetMemoryOwner();
  CHECK(row_set_mem_owner);
  auto result_set = std::make_unique<ResultSet>(
      target_infos, ExecutorDeviceType::CPU, query_mem_desc, row_set_mem_owner, executor);
  if (query_mem_desc.getEntryCount()) {
    auto storage = result_set->allocateStorage(serialized_rows.target_init_vals);
    auto storage_buff = storage->getUnderlyingBuffer();
    memcpy(storage_buff, serialized_rows.buffer.data(), serialized_rows.buffer.size());
    result_set->unserializeCountDistinctColumns(serialized_rows);
  }
  result_set->separate_varlen_storage_valid_ = true;
  result_set->serialized_varlen_buffer_.emplace_back(
      std::move(serialized_rows.varlen_buffer));
  return result_set;
}

void ResultSet::unserializeCountDistinctColumns(const TSerializedRows& serialized_rows) {
  for (const auto& count_distinct_set : serialized_rows.count_distinct_sets) {
    CHECK(count_distinct_set.remote_ptr);
    switch (count_distinct_set.type) {
      case TCountDistinctImplType::Bitmap: {
        CHECK(!count_distinct_set.storage.bitmap.empty());
        const auto bitmap_byte_sz = count_distinct_set.storage.bitmap.size();
        auto count_distinct_buffer = static_cast<int8_t*>(checked_malloc(bitmap_byte_sz));
        memcpy(
            count_distinct_buffer, &count_distinct_set.storage.bitmap[0], bitmap_byte_sz);
        row_set_mem_owner_->addCountDistinctBuffer(
            count_distinct_buffer, bitmap_byte_sz, true);
        storage_->addCountDistinctSetPointerMapping(
            count_distinct_set.remote_ptr,
            reinterpret_cast<int64_t>(count_distinct_buffer));
        break;
      }
      case TCountDistinctImplType::StdSet: {
        auto count_distinct_sparse_set =
            new std::set<int64_t>(count_distinct_set.storage.sparse_set);
        row_set_mem_owner_->addCountDistinctSet(count_distinct_sparse_set);
        storage_->addCountDistinctSetPointerMapping(
            count_distinct_set.remote_ptr,
            reinterpret_cast<int64_t>(count_distinct_sparse_set));
        break;
      }
      default:
        CHECK(false);
    }
  }
  // Just because set is empty doesnt mean there are not zero results to be recreated
  // need to check for a column containg count distinct values
  // look through the resultset to check if there are actualy any distinct columns before
  // doing this
  for (size_t target_idx = 0; target_idx < storage_->targets_.size(); ++target_idx) {
    const auto& target_info = storage_->targets_[target_idx];
    if (is_distinct_target(target_info)) {
      fixupCountDistinctPointers();
      break;
    }
  }
}

void ResultSet::fixupCountDistinctPointers() {
  if (query_mem_desc_.getEntryCount() > 100000) {
    const size_t worker_count = cpu_threads();
    std::vector<std::future<void>> fixup_threads;
    for (size_t
             i = 0,
             start_entry = 0,
             stride = (query_mem_desc_.getEntryCount() + worker_count - 1) / worker_count;
         i < worker_count && start_entry < query_mem_desc_.getEntryCount();
         ++i, start_entry += stride) {
      const auto end_entry =
          std::min(start_entry + stride, query_mem_desc_.getEntryCount());
      fixup_threads.push_back(std::async(std::launch::async,
                                         [this](const size_t start, const size_t end) {
                                           for (size_t i = start; i < end; ++i) {
                                             getRowAt(i, false, false, true);
                                           }
                                         },
                                         start_entry,
                                         end_entry));
    }
    for (auto& child : fixup_threads) {
      child.get();
    }
  } else {
    for (size_t i = 0; i < query_mem_desc_.getEntryCount(); ++i) {
      getRowAt(i, false, false, true);
    }
  }
  decltype(storage_->count_distinct_sets_mapping_)().swap(
      storage_->count_distinct_sets_mapping_);
}
