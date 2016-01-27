#ifdef HAVE_CALCITE
#include "RelAlgAbstractInterpreter.h"

#include <glog/logging.h>

#include <unordered_map>

ScanBufferDesc::ScanBufferDesc() : td_(nullptr) {
}

ScanBufferDesc::ScanBufferDesc(const TableDescriptor* td) : td_(td) {
  CHECK(td_);
}

namespace {

std::string node_id(const rapidjson::Value& ra_node) {
  CHECK(ra_node.IsObject());
  const auto id_it = ra_node.FindMember("id");
  CHECK(id_it != ra_node.MemberEnd());
  const auto& id = id_it->value;
  CHECK(id.IsString());
  return id.GetString();
}

class RaAbstractInterp {
 public:
  RaAbstractInterp(const rapidjson::Value& query_ast, const Catalog_Namespace::Catalog& cat)
      : query_ast_(query_ast), cat_(cat) {}

  LoweringInfo run() {
    const auto rels_mem_it = query_ast_.FindMember("rels");
    CHECK(rels_mem_it != query_ast_.MemberEnd());
    const auto& rels = rels_mem_it->value;
    CHECK(rels.IsArray());
    for (auto rels_it = rels.Begin(); rels_it != rels.End(); ++rels_it) {
      const auto& crt_node = *rels_it;
      CHECK(crt_node.IsObject());
      const auto rel_op_it = crt_node.FindMember("relOp");
      CHECK(rel_op_it != crt_node.MemberEnd());
      if (rel_op_it->value.GetString() == std::string("LogicalTableScan")) {
        dispatchTableScan(crt_node);
      } else {
        CHECK(false);
      }
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

  void dispatchTableScan(const rapidjson::Value& scan_ra) {
    CHECK(scan_ra.IsObject());
    const auto td = getTableFromScanNode(scan_ra);
    CHECK(td);
    ra_id_to_scan_buffer_.insert(std::make_pair(node_id(scan_ra), ScanBufferDesc(td)));
  }

  const rapidjson::Value& query_ast_;
  const Catalog_Namespace::Catalog& cat_;
  std::unordered_map<std::string, ScanBufferDesc> ra_id_to_scan_buffer_;
};

}  // namespace

LoweringInfo ra_interpret(const rapidjson::Value& query_ast, const Catalog_Namespace::Catalog& cat) {
  RaAbstractInterp interp(query_ast, cat);
  return interp.run();
}
#endif  // HAVE_CALCITE
