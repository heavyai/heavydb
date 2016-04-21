#include "InputMetadata.h"
#include "GroupByAndAggregate.h"

namespace {

bool uses_int_meta(const SQLTypeInfo& col_ti) {
  return col_ti.is_integer() || col_ti.is_decimal() || col_ti.is_time() || col_ti.is_boolean() ||
         (col_ti.is_string() && col_ti.get_compression() == kENCODING_DICT);
}

// TODO(alex): Placeholder, provide an efficient implementation for this
std::map<int, ChunkMetadata> synthesize_metadata(const ResultRows* rows) noexcept {
  rows->moveToBegin();
  std::vector<std::unique_ptr<Encoder>> dummy_encoders;
  for (size_t i = 0; i < rows->colCount(); ++i) {
    const auto& col_ti = rows->getColType(i);
    dummy_encoders.emplace_back(Encoder::Create(nullptr, col_ti));
  }
  while (true) {
    auto crt_row = rows->getNextRow(false, false);
    if (crt_row.empty()) {
      break;
    }
    for (size_t i = 0; i < rows->colCount(); ++i) {
      const auto& col_ti = rows->getColType(i);
      const auto& col_val = crt_row[i];
      const auto scalar_col_val = boost::get<ScalarTargetValue>(&col_val);
      CHECK(scalar_col_val);
      if (uses_int_meta(col_ti)) {
        const auto i64_p = boost::get<int64_t>(scalar_col_val);
        CHECK(i64_p);
        dummy_encoders[i]->updateStats(*i64_p, *i64_p == inline_int_null_val(col_ti));
      } else {
        CHECK(col_ti.is_fp());
        const auto double_p = boost::get<double>(scalar_col_val);
        CHECK(double_p);
        dummy_encoders[i]->updateStats(*double_p, *double_p == inline_fp_null_val(col_ti));
      }
    }
  }
  rows->moveToBegin();
  std::map<int, ChunkMetadata> metadata_map;
  for (size_t i = 0; i < rows->colCount(); ++i) {
    const auto it_ok = metadata_map.emplace(i, dummy_encoders[i]->getMetadata(rows->getColType(i)));
    CHECK(it_ok.second);
  }
  return metadata_map;
}

Fragmenter_Namespace::TableInfo synthesize_table_info(const ResultRows* rows) noexcept {
  std::deque<Fragmenter_Namespace::FragmentInfo> result;
  const size_t row_count = rows->rowCount();
  if (row_count) {
    result.resize(1);
    auto& fragment = result.front();
    fragment.fragmentId = 0;
    fragment.numTuples = row_count;
    fragment.deviceIds.resize(3);
    fragment.chunkMetadataMap = synthesize_metadata(rows);
  }
  Fragmenter_Namespace::TableInfo table_info;
  table_info.fragments = result;
  table_info.numTuples = row_count;
  return table_info;
}

}  // namespace

std::vector<Fragmenter_Namespace::TableInfo> get_table_infos(const std::vector<InputDescriptor>& input_descs,
                                                             const Catalog_Namespace::Catalog& cat,
                                                             const TemporaryTables& temporary_tables) noexcept {
  std::vector<Fragmenter_Namespace::TableInfo> table_infos;
  for (const auto& input_desc : input_descs) {
    if (input_desc.getSourceType() == InputSourceType::RESULT) {
      const int temp_table_id = input_desc.getTableId();
      CHECK_LT(temp_table_id, 0);
      const auto it = temporary_tables.find(temp_table_id);
      CHECK(it != temporary_tables.end());
      table_infos.push_back(synthesize_table_info(it->second));
      continue;
    }
    CHECK(input_desc.getSourceType() == InputSourceType::TABLE);
    const auto table_descriptor = cat.getMetadataForTable(input_desc.getTableId());
    CHECK(table_descriptor);
    const auto fragmenter = table_descriptor->fragmenter;
    CHECK(fragmenter);
    table_infos.push_back(fragmenter->getFragmentsForQuery());
  }
  return table_infos;
}
