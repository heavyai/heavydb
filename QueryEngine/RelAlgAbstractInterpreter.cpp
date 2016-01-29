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

// Checked json field retrieval.
const rapidjson::Value& field(const rapidjson::Value& obj, const char field[]) noexcept {
  CHECK(obj.IsObject());
  const auto field_it = obj.FindMember(field);
  CHECK(field_it != obj.MemberEnd());
  return field_it->value;
}

const int64_t json_i64(const rapidjson::Value& obj) noexcept {
  CHECK(obj.IsInt64());
  return obj.GetInt64();
}

const std::string json_str(const rapidjson::Value& obj) noexcept {
  CHECK(obj.IsString());
  return obj.GetString();
}

const bool json_bool(const rapidjson::Value& obj) noexcept {
  CHECK(obj.IsBool());
  return obj.GetBool();
}

const double json_double(const rapidjson::Value& obj) noexcept {
  CHECK(obj.IsDouble());
  return obj.GetDouble();
}

unsigned node_id(const rapidjson::Value& ra_node) noexcept {
  const auto& id = field(ra_node, "id");
  return std::stoi(json_str(id));
}

RexAbstractInput* parse_abstract_input(const rapidjson::Value& expr) noexcept {
  const auto& input = field(expr, "input");
  return new RexAbstractInput(json_i64(input));
}

RexLiteral* parse_literal(const rapidjson::Value& expr) noexcept {
  CHECK(expr.IsObject());
  const auto& literal = field(expr, "literal");
  const auto type = to_sql_type(json_str(field(expr, "type")));
  const auto scale = json_i64(field(expr, "scale"));
  const auto precision = json_i64(field(expr, "precision"));
  const auto type_scale = json_i64(field(expr, "type_scale"));
  const auto type_precision = json_i64(field(expr, "type_precision"));
  switch (type) {
    case kDECIMAL:
      return new RexLiteral(json_i64(literal), type, scale, precision, type_scale, type_precision);
    case kDOUBLE:
      return new RexLiteral(json_double(literal), type, scale, precision, type_scale, type_precision);
    case kTEXT:
      return new RexLiteral(json_str(literal), type, scale, precision, type_scale, type_precision);
    case kBOOLEAN:
      return new RexLiteral(json_bool(literal), type, scale, precision, type_scale, type_precision);
    case kNULLT:
      return new RexLiteral();
    default:
      CHECK(false);
  }
  CHECK(false);
  return nullptr;
}

Rex* parse_expr(const rapidjson::Value& expr);

RexOperator* parse_operator(const rapidjson::Value& expr) {
  const auto op = to_sql_op(json_str(field(expr, "op")));
  const auto& operators_json_arr = field(expr, "operands");
  CHECK(operators_json_arr.IsArray());
  std::vector<const RexScalar*> operands;
  for (auto operators_json_arr_it = operators_json_arr.Begin(); operators_json_arr_it != operators_json_arr.End();
       ++operators_json_arr_it) {
    const auto operand = dynamic_cast<RexScalar*>(parse_expr(*operators_json_arr_it));
    CHECK(operand);
    operands.push_back(operand);
  }
  return new RexOperator(op, operands);
}

Rex* parse_expr(const rapidjson::Value& expr) {
  CHECK(expr.IsObject());
  if (expr.IsObject() && expr.HasMember("input")) {
    return parse_abstract_input(expr);
  }
  if (expr.IsObject() && expr.HasMember("literal")) {
    return parse_literal(expr);
  }
  if (expr.IsObject() && expr.HasMember("op")) {
    return parse_operator(expr);
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
    const auto& rels = field(query_ast_, "rels");
    CHECK(rels.IsArray());
    for (auto rels_it = rels.Begin(); rels_it != rels.End(); ++rels_it) {
      const auto& crt_node = *rels_it;
      const auto id = node_id(crt_node);
      CHECK_EQ(static_cast<size_t>(id), nodes_.size());
      CHECK(crt_node.IsObject());
      const auto rel_op = json_str(field(crt_node, "relOp"));
      if (rel_op == std::string("LogicalTableScan")) {
        nodes_.push_back(dispatchTableScan(crt_node));
      } else if (rel_op == std::string("LogicalProject")) {
        nodes_.push_back(dispatchProject(crt_node));
      } else if (rel_op == std::string("LogicalFilter")) {
        nodes_.push_back(dispatchFilter(crt_node));
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
    const auto& exprs_json = field(proj_ra, "exprs");
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

  RelFilter* dispatchFilter(const rapidjson::Value& filter_ra) {
    return new RelFilter(parse_operator(field(filter_ra, "condition")));
  }

  const TableDescriptor* getTableFromScanNode(const rapidjson::Value& scan_ra) const {
    const auto& table_json = field(scan_ra, "table");
    CHECK(table_json.IsArray());
    CHECK_EQ(unsigned(3), table_json.Size());
    const auto td = cat_.getMetadataForTable(table_json[2].GetString());
    CHECK(td);
    return td;
  }

  std::vector<std::string> getFieldNamesFromScanNode(const rapidjson::Value& scan_ra) const {
    const auto& fields_json = field(scan_ra, "fieldNames");
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
