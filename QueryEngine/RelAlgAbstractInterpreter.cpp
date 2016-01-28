#ifdef HAVE_CALCITE
#include "RelAlgAbstractInterpreter.h"

#include <glog/logging.h>

#include <string>
#include <unordered_map>

ScanBufferDesc::ScanBufferDesc() : td_(nullptr) {
}

ScanBufferDesc::ScanBufferDesc(const TableDescriptor* td) : td_(td) {
  CHECK(td_);
}

namespace {

unsigned node_id(const rapidjson::Value& ra_node) noexcept {
  CHECK(ra_node.IsObject());
  const auto id_it = ra_node.FindMember("id");
  CHECK(id_it != ra_node.MemberEnd());
  const auto& id = id_it->value;
  CHECK(id.IsString());
  return std::stoi(id.GetString());
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
      const auto id = node_id(crt_node);
      CHECK_EQ(static_cast<size_t>(id), nodes_.size());
      CHECK(crt_node.IsObject());
      const auto rel_op_it = crt_node.FindMember("relOp");
      CHECK(rel_op_it != crt_node.MemberEnd());
      if (rel_op_it->value.GetString() == std::string("LogicalTableScan")) {
        nodes_.push_back(dispatchTableScan(crt_node));
      } else {
        CHECK(false);
      }
      CHECK(false);
    }
    return {};
  }

 private:
  RelAlgScan* dispatchTableScan(const rapidjson::Value& scan_ra) {
    CHECK(scan_ra.IsObject());
    const auto td = getTableFromScanNode(scan_ra);
    const auto field_names = getFieldNamesFromScanNode(scan_ra);
    return new RelAlgScan(td, field_names);
  }

  const TableDescriptor* getTableFromScanNode(const rapidjson::Value& scan_ra) const {
    const auto& table_info = scan_ra["table"];
    CHECK(table_info.IsArray());
    CHECK_EQ(unsigned(3), table_info.Size());
    const auto td = cat_.getMetadataForTable(table_info[2].GetString());
    CHECK(td);
    return td;
  }

  std::vector<std::string> getFieldNamesFromScanNode(const rapidjson::Value& scan_ra) const {
    const auto it = scan_ra.FindMember("fieldNames");
    CHECK(it != scan_ra.MemberEnd());
    const auto& fields_json = it->value;
    CHECK(fields_json.IsArray());
    std::vector<std::string> fields;
    for (auto fields_json_it = fields_json.Begin(); fields_json_it != fields_json.End(); ++fields_json_it) {
      CHECK(fields_json_it->IsString());
      fields.emplace_back(fields_json_it->GetString());
    }
    return fields;
  }

  const rapidjson::Value& query_ast_;
  const Catalog_Namespace::Catalog& cat_;
  std::vector<RelAlgNode*> nodes_;
};

}  // namespace

LoweringInfo ra_interpret(const rapidjson::Value& query_ast, const Catalog_Namespace::Catalog& cat) {
  RaAbstractInterp interp(query_ast, cat);
  return interp.run();
}
#endif  // HAVE_CALCITE
