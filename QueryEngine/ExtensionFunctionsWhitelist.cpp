#include "ExtensionFunctionsWhitelist.h"
#include "JsonAccessors.h"

ExtensionFunction* ExtensionFunctionsWhitelist::get(const std::string& name) {
  const auto it = functions_.find(name);
  if (it == functions_.end()) {
    return nullptr;
  }
  return &it->second;
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
    const auto it_ok = functions_.emplace(name, ExtensionFunction(args, ret));
    CHECK(it_ok.second);
  }
}

std::unordered_map<std::string, ExtensionFunction> ExtensionFunctionsWhitelist::functions_;
