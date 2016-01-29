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

static RexAbstractInput* parse_abstract_input(const rapidjson::Value& expr) {
  CHECK(expr.IsObject());
  const auto input_field_it = expr.FindMember("input");
  CHECK(input_field_it != expr.MemberEnd());
  const auto& input_field_json = input_field_it->value;
  CHECK(input_field_json.IsInt());
  return new RexAbstractInput(input_field_json.GetInt());
}

static Rex* parse_expr(const rapidjson::Value& expr) {
  CHECK(expr.IsObject());
  if (expr.IsObject() && expr.HasMember("input")) {
    return parse_abstract_input(expr);
  }
  CHECK(false);
  return nullptr;
}

std::vector<std::string> strings_from_json_array(const rapidjson::Value& json_str_arr) {
  std::vector<std::string> fields;
  for (auto json_str_arr_it = json_str_arr.Begin(); json_str_arr_it != json_str_arr.End(); ++json_str_arr_it) {
    CHECK(json_str_arr_it->IsString());
    fields.emplace_back(json_str_arr_it->GetString());
  }
  return fields;
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
      } else if (rel_op_it->value.GetString() == std::string("LogicalProject")) {
        nodes_.push_back(dispatchProject(crt_node));
      } else {
        CHECK(false);
      }
    }
    return {};
  }

 private:
  RelScan* dispatchTableScan(const rapidjson::Value& scan_ra) {
    CHECK(scan_ra.IsObject());
    const auto td = getTableFromScanNode(scan_ra);
    const auto field_names = getFieldNamesFromScanNode(scan_ra);
    return new RelScan(td, field_names);
  }

  RelProject* dispatchProject(const rapidjson::Value& proj_ra) {
    const auto exprs_field_it = proj_ra.FindMember("exprs");
    CHECK(exprs_field_it != proj_ra.MemberEnd());
    const auto& exprs_json = exprs_field_it->value;
    CHECK(exprs_json.IsArray());
    std::vector<const RexScalar*> exprs;
    for (auto exprs_json_it = exprs_json.Begin(); exprs_json_it != exprs_json.End(); ++exprs_json_it) {
      const auto scalar_expr = dynamic_cast<const RexScalar*>(parse_expr(*exprs_json_it));
      CHECK(scalar_expr);
      exprs.push_back(scalar_expr);
    }
    const auto fields_field_it = proj_ra.FindMember("fields");
    CHECK(fields_field_it != proj_ra.MemberEnd());
    return new RelProject(exprs, strings_from_json_array(fields_field_it->value));
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
    return strings_from_json_array(fields_json);
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
