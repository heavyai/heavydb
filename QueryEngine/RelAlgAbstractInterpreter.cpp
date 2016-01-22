#include "RelAlgAbstractInterpreter.h"

#include <glog/logging.h>

#include <unordered_map>

ScanBufferDesc::ScanBufferDesc() : td_(nullptr) {
}

ScanBufferDesc::ScanBufferDesc(const TableDescriptor* td) : td_(td) {
  CHECK(td_);
}

namespace {

class RaAbstractInterp {
 public:
  RaAbstractInterp(const rapidjson::Value& query_ast, const Catalog_Namespace::Catalog& cat)
      : query_ast_(query_ast), cat_(cat) {}

  LoweringInfo run() {
    const auto& rels = query_ast_["rels"];
    CHECK(rels.IsArray());
    for (auto rels_it = rels.Begin(); rels_it != rels.End(); ++rels_it) {
      CHECK(false);
    }
    return {};
  }

 private:
  const TableDescriptor* getTableFromScanNode(const rapidjson::Value& scan_ra) const {
    const auto& table_info = scan_ra["table"];
    CHECK(table_info.IsArray());
    CHECK_EQ(unsigned(3), table_info.Size());
    const auto td = cat_.getMetadataForTable(table_info[2].GetString());
    CHECK(td);
    return td;
  }

  const rapidjson::Value& query_ast_;
  const Catalog_Namespace::Catalog& cat_;
  std::unordered_map<int, ScanBufferDesc> ra_id_to_scan_buffer_;
};

}  // namespace

LoweringInfo ra_interpret(const rapidjson::Value& query_ast, const Catalog_Namespace::Catalog& cat) {
  RaAbstractInterp interp(query_ast, cat);
  return interp.run();
}
