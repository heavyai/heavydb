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

std::vector<Fragmenter_Namespace::TableInfo> get_table_infos(const std::vector<ScanDescriptor>& scan_ids,
                                                             const Catalog_Namespace::Catalog& cat) {
  std::vector<Fragmenter_Namespace::TableInfo> table_infos;
  for (const auto& scan_id : scan_ids) {
    if (scan_id.getSourceType() == InputSourceType::RESULT) {
      table_infos.push_back(synthesize_table_info(scan_id.getResultRows()));
      continue;
    }
    CHECK(scan_id.getSourceType() == InputSourceType::TABLE);
    const auto table_descriptor = cat.getMetadataForTable(scan_id.getTableId());
    CHECK(table_descriptor);
    const auto fragmenter = table_descriptor->fragmenter;
    CHECK(fragmenter);
    table_infos.push_back(fragmenter->getFragmentsForQuery());
  }
  return table_infos;
}
