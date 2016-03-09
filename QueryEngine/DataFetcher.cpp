#include "DataFetcher.h"
#include "GroupByAndAggregate.h"

std::deque<Fragmenter_Namespace::FragmentInfo> get_fragments_meta(const ResultRows* rows) {
  std::deque<Fragmenter_Namespace::FragmentInfo> result(1);
  auto& fragment = result.front();
  fragment.fragmentId = 0;
  fragment.numTuples = rows->rowCount();
  fragment.deviceIds.resize(3);
  CHECK(false);
  return result;
}

std::vector<Fragmenter_Namespace::QueryInfo> get_query_infos(const std::vector<ScanDescriptor>& scan_ids,
                                                             const Catalog_Namespace::Catalog& cat) {
  std::vector<Fragmenter_Namespace::QueryInfo> query_infos;
  {
    for (const auto& scan_id : scan_ids) {
      const auto table_descriptor = cat.getMetadataForTable(scan_id.getTableId());
      CHECK(table_descriptor);
      const auto fragmenter = table_descriptor->fragmenter;
      CHECK(fragmenter);
      query_infos.push_back(fragmenter->getFragmentsForQuery());
    }
  }
  return query_infos;
}
