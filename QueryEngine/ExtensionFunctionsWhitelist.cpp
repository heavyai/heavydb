#include "ExtensionFunctionsWhitelist.h"
#include "JsonAccessors.h"

#include "../Shared/StringTransform.h"

#include <boost/algorithm/string/join.hpp>

std::vector<ExtensionFunction>* ExtensionFunctionsWhitelist::get(const std::string& name) {
  const auto it = functions_.find(to_upper(name));
  if (it == functions_.end()) {
    return nullptr;
  }
  return &it->second;
}

namespace {

std::string serialize_type(const ExtArgumentType type) {
  switch (type) {
    case ExtArgumentType::Int16:
      return "i16";
    case ExtArgumentType::Int32:
      return "i32";
    case ExtArgumentType::Int64:
      return "i64";
    case ExtArgumentType::Float:
      return "float";
    case ExtArgumentType::Double:
      return "double";
    default:
      CHECK(false);
  }
  CHECK(false);
  return "";
}

}  // namespace

std::vector<std::string> ExtensionFunctionsWhitelist::getLLVMDeclarations() {
  std::vector<std::string> declarations;
  for (const auto& kv : functions_) {
    const auto& signatures = kv.second;
    CHECK(!signatures.empty());
    for (const auto& signature : kv.second) {
      std::string decl_prefix{"declare " + serialize_type(signature.getRet()) + " @" + signature.getName()};
      std::vector<std::string> arg_strs;
      for (const auto arg : signature.getArgs()) {
        arg_strs.push_back(serialize_type(arg));
      }
      declarations.push_back(decl_prefix + "(" + boost::algorithm::join(arg_strs, ", ") + ");");
    }
  }
  return declarations;
}

namespace {

ExtArgumentType deserialize_type(const std::string& type_name) {
  if (type_name == "i16") {
    return ExtArgumentType::Int16;
  }
  if (type_name == "i32") {
    return ExtArgumentType::Int32;
  }
  if (type_name == "i64") {
    return ExtArgumentType::Int64;
  }
  if (type_name == "float") {
    return ExtArgumentType::Float;
  }
  if (type_name == "double") {
    return ExtArgumentType::Double;
  }
  CHECK(false);
  return ExtArgumentType::Int16;
}

}  // namespace

// Valid json_func_sigs example:
// [
//    {
//       "name":"sum",
//       "ret":"i32",
//       "args":[
//          "i32",
//          "i32"
//       ]
//    }
// ]
void ExtensionFunctionsWhitelist::add(const std::string& json_func_sigs) {
  rapidjson::Document func_sigs;
  func_sigs.Parse(json_func_sigs.c_str());
  CHECK(func_sigs.IsArray());
  for (auto func_sigs_it = func_sigs.Begin(); func_sigs_it != func_sigs.End(); ++func_sigs_it) {
    CHECK(func_sigs_it->IsObject());
    const auto name = json_str(field(*func_sigs_it, "name"));
    const auto ret = deserialize_type(json_str(field(*func_sigs_it, "ret")));
    std::vector<ExtArgumentType> args;
    const auto& args_serialized = field(*func_sigs_it, "args");
    CHECK(args_serialized.IsArray());
    for (auto args_serialized_it = args_serialized.Begin(); args_serialized_it != args_serialized.End();
         ++args_serialized_it) {
      args.push_back(deserialize_type(json_str(*args_serialized_it)));
    }
    functions_[to_upper(name)].emplace_back(name, args, ret);
  }
}

std::unordered_map<std::string, std::vector<ExtensionFunction>> ExtensionFunctionsWhitelist::functions_;
