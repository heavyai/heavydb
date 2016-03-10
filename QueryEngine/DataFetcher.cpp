#include "DataFetcher.h"
#include "GroupByAndAggregate.h"

namespace {

Fragmenter_Namespace::TableInfo synthesize_table_info(const ResultRows* rows) {
  std::deque<Fragmenter_Namespace::FragmentInfo> result(1);
  auto& fragment = result.front();
  fragment.fragmentId = 0;
  fragment.numTuples = rows->rowCount();
  fragment.deviceIds.resize(3);
  Fragmenter_Namespace::TableInfo table_info;
  table_info.fragments = result;
  table_info.numTuples = fragment.numTuples;
  return table_info;
}

}  // namespace

std::vector<Fragmenter_Namespace::TableInfo> get_table_infos(const std::vector<ScanDescriptor>& scan_descs,
                                                             const Catalog_Namespace::Catalog& cat,
                                                             const TemporaryTables& temporary_tables) {
  std::vector<Fragmenter_Namespace::TableInfo> table_infos;
  for (const auto& scan_desc : scan_descs) {
    if (scan_desc.getSourceType() == InputSourceType::RESULT) {
      const int temp_table_id = scan_desc.getTableId();
      CHECK_LT(temp_table_id, 0);
      const auto it = temporary_tables.find(temp_table_id);
      CHECK(it != temporary_tables.end());
      table_infos.push_back(synthesize_table_info(it->second));
      continue;
    }
    CHECK(scan_desc.getSourceType() == InputSourceType::TABLE);
    const auto table_descriptor = cat.getMetadataForTable(scan_desc.getTableId());
    CHECK(table_descriptor);
    const auto fragmenter = table_descriptor->fragmenter;
    CHECK(fragmenter);
    table_infos.push_back(fragmenter->getFragmentsForQuery());
  }
  return table_infos;
}
