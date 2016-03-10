#include "DataFetcher.h"
#include "GroupByAndAggregate.h"

namespace {

Fragmenter_Namespace::QueryInfo synthesize_query_info(const ResultRows* rows) {
  std::deque<Fragmenter_Namespace::FragmentInfo> result(1);
  auto& fragment = result.front();
  fragment.fragmentId = 0;
  fragment.numTuples = rows->rowCount();
  fragment.deviceIds.resize(3);
  Fragmenter_Namespace::QueryInfo query_info;
  query_info.fragments = result;
  query_info.numTuples = fragment.numTuples;
  return query_info;
}

}  // namespace

std::vector<Fragmenter_Namespace::QueryInfo> get_query_infos(const std::vector<ScanDescriptor>& scan_ids,
                                                             const Catalog_Namespace::Catalog& cat) {
  std::vector<Fragmenter_Namespace::QueryInfo> query_infos;
  for (const auto& scan_id : scan_ids) {
    if (scan_id.getSourceType() == InputSourceType::RESULT) {
      query_infos.push_back(synthesize_query_info(scan_id.getResultRows()));
      continue;
    }
    CHECK(scan_id.getSourceType() == InputSourceType::TABLE);
    const auto table_descriptor = cat.getMetadataForTable(scan_id.getTableId());
    CHECK(table_descriptor);
    const auto fragmenter = table_descriptor->fragmenter;
    CHECK(fragmenter);
    query_infos.push_back(fragmenter->getFragmentsForQuery());
  }
  return query_infos;
}
