/*
 * Copyright 2022 Intel Corporation.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "CalciteJNI.h"

#include "Logger/Logger.h"
#include "OSDependent/omnisci_path.h"

#include <jni.h>

extern bool g_enable_watchdog;

namespace {

void createJVM(JavaVM** jvm, JNIEnv** env, const std::vector<std::string>& jvm_opts) {
  JavaVMInitArgs vm_args;
  auto options = std::make_unique<JavaVMOption[]>(jvm_opts.size());
  for (size_t i = 0; i < jvm_opts.size(); ++i) {
    options[i].optionString = const_cast<char*>(jvm_opts[i].c_str());
  }
  vm_args.version = JNI_VERSION_1_8;
  vm_args.nOptions = 1;
  vm_args.options = options.get();
  vm_args.ignoreUnrecognized = false;
  if (JNI_CreateJavaVM(jvm, (void**)env, &vm_args) != JNI_OK) {
    throw std::runtime_error("Couldn't initialize JVM.");
  }
}

}  // namespace

class CalciteJNI::Impl {
 public:
  Impl(const std::string& udf_filename, size_t calcite_max_mem_mb) {
    // Initialize JVM.
    auto root_abs_path = omnisci::get_root_abs_path();
    std::string class_path_arg = "-Djava.class.path=" + root_abs_path +
                                 "/bin/calcite-1.0-SNAPSHOT-jar-with-dependencies.jar";
    std::string max_mem_arg = "-Xmx" + std::to_string(calcite_max_mem_mb) + "m";
    createJVM(&jvm_, &env_, {class_path_arg, max_mem_arg});

    // Create CalciteServerHandler object.
    createCalciteServerHandler(udf_filename);

    // Prepare references to some Java classes and methods we will use for processing.
    findTQueryParsingOption();
    findTOptimizationOption();
    findTPlanResult();
    findTExtArgumentType();
    findTOutputBufferSizeType();
    findTUserDefinedTableFunction();
    findInvalidParseRequest();
    findArrayList();
    findHashMap();
  }

  ~Impl() { jvm_->DestroyJavaVM(); }

  TPlanResult process(const std::string& user,
                      const std::string& db_name,
                      const std::string& sql_string,
                      const std::string& schema_json,
                      const std::string& session_id,
                      const std::vector<TFilterPushDownInfo>& filter_push_down_info,
                      const bool legacy_syntax,
                      const bool is_explain,
                      const bool is_view_optimize) {
    jstring arg_user = env_->NewStringUTF(user.c_str());
    jstring arg_session = env_->NewStringUTF(session_id.c_str());
    jstring arg_catalog = env_->NewStringUTF(db_name.c_str());
    jstring arg_query = env_->NewStringUTF(sql_string.c_str());
    jobject arg_parsing_options = env_->NewObject(parsing_opts_cls_,
                                                  parsing_opts_ctor_,
                                                  (jboolean)legacy_syntax,
                                                  (jboolean)is_explain,
                                                  /*check_privileges=*/(jboolean)(false));
    if (!arg_parsing_options) {
      throw std::runtime_error("cannot create TQueryParsingOption object");
    }
    jobject arg_filter_push_down_info =
        env_->NewObject(array_list_cls_, array_list_ctor_);
    if (!filter_push_down_info.empty()) {
      throw std::runtime_error("Filter pushdown info is not supported in Calcite JNI.");
    }
    jobject arg_optimization_options = env_->NewObject(optimization_opts_cls_,
                                                       optimization_opts_ctor_,
                                                       (jboolean)is_view_optimize,
                                                       (jboolean)g_enable_watchdog,
                                                       arg_filter_push_down_info);
    jobject arg_restriction = nullptr;
    jstring arg_schema = env_->NewStringUTF(schema_json.c_str());

    jobject java_res = env_->CallObjectMethod(handler_obj_,
                                              handler_process_,
                                              arg_user,
                                              arg_session,
                                              arg_catalog,
                                              arg_query,
                                              arg_parsing_options,
                                              arg_optimization_options,
                                              arg_restriction,
                                              arg_schema);
    if (!java_res) {
      if (env_->ExceptionCheck() == JNI_FALSE) {
        throw std::runtime_error(
            "CalciteServerHandler::process call failed for unknown reason\n  Query: " +
            sql_string + "\n  Schema: " + schema_json);
      } else {
        jthrowable e = env_->ExceptionOccurred();
        CHECK(e);
        throw std::invalid_argument(readStringField(e, invalid_parse_req_why_));
      }
    }

    TPlanResult res;
    res.plan_result = readStringField(java_res, plan_result_plan_result_);
    return res;
  }

  std::string getExtensionFunctionWhitelist() {
    jstring java_res =
        (jstring)env_->CallObjectMethod(handler_obj_, handler_get_ext_fn_list_);
    return convertJavaString(java_res);
  }

  std::string getUserDefinedFunctionWhitelist() {
    jstring java_res =
        (jstring)env_->CallObjectMethod(handler_obj_, handler_get_udf_list_);
    return convertJavaString(java_res);
  }

  std::string getRuntimeExtensionFunctionWhitelist() {
    jstring java_res =
        (jstring)env_->CallObjectMethod(handler_obj_, handlhandler_get_rt_fn_list_);
    return convertJavaString(java_res);
  }

  void setRuntimeExtensionFunctions(const std::vector<TUserDefinedFunction>& udfs,
                                    const std::vector<TUserDefinedTableFunction>& udtfs,
                                    bool is_runtime) {
    jobject udfs_list = env_->NewObject(array_list_cls_, array_list_ctor_);
    if (!udfs.empty()) {
      std::runtime_error("TUserDefinedFunction is not yet supported in Calcite JNI");
    }

    jobject udtfs_list = env_->NewObject(array_list_cls_, array_list_ctor_);
    for (auto& udtf : udtfs) {
      env_->CallVoidMethod(
          udtfs_list, array_list_add_, convertTUserDefinedTableFunction(udtf));
    }

    env_->CallVoidMethod(
        handler_obj_, handler_set_rt_fns_, udfs_list, udtfs_list, (jboolean)is_runtime);
    if (env_->ExceptionCheck() != JNI_FALSE) {
      throw std::runtime_error("Failed Java call to setRuntimeExtensionFunctions");
    }
  }

 private:
  void createCalciteServerHandler(const std::string& udf_filename) {
    jclass handler_cls = env_->FindClass("com/mapd/parser/server/CalciteServerHandler");
    if (!handler_cls) {
      throw std::runtime_error("cannot find Java class CalciteServerHandler");
    }

    jmethodID handler_ctor =
        env_->GetMethodID(handler_cls,
                          "<init>",
                          "(ILjava/lang/String;Ljava/lang/String;Lcom/mapd/common/"
                          "SockTransportProperties;Ljava/lang/String;)V");
    if (!handler_ctor) {
      throw std::runtime_error("cannot find CalciteServerHandler ctor");
    }

    int port = -1;
    jstring data_dir = env_->NewStringUTF("");
    auto root_abs_path = omnisci::get_root_abs_path();
    std::string ext_ast_path = root_abs_path + "/QueryEngine/ExtensionFunctions.ast";
    jstring ext_ast_file = env_->NewStringUTF(ext_ast_path.c_str());
    jobject sock_transport_prop = nullptr;
    jstring udf_ast_file = env_->NewStringUTF(udf_filename.c_str());
    handler_obj_ = env_->NewObject(handler_cls,
                                   handler_ctor,
                                   port,
                                   data_dir,
                                   ext_ast_file,
                                   sock_transport_prop,
                                   udf_ast_file);
    if (!handler_obj_) {
      throw std::runtime_error("cannot create CalciteServerHandler object");
    }

    // Find 'CalciteServerHandler::process' method.
    handler_process_ = env_->GetMethodID(
        handler_cls,
        "process",
        "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Lcom/"
        "omnisci/thrift/calciteserver/TQueryParsingOption;Lcom/omnisci/thrift/"
        "calciteserver/TOptimizationOption;Lcom/omnisci/thrift/calciteserver/"
        "TRestriction;Ljava/lang/String;)Lcom/omnisci/thrift/calciteserver/TPlanResult;");
    if (!handler_process_) {
      throw std::runtime_error("cannot find CalciteServerHandler::process method");
    }

    // Find 'CalciteServerHandler::getExtensionFunctionWhitelist' method.
    handler_get_ext_fn_list_ = env_->GetMethodID(
        handler_cls, "getExtensionFunctionWhitelist", "()Ljava/lang/String;");
    if (!handler_get_ext_fn_list_) {
      throw std::runtime_error(
          "cannot find CalciteServerHandler::getExtensionFunctionWhitelist method");
    }

    // Find 'CalciteServerHandler::getUserDefinedFunctionWhitelist' method.
    handler_get_udf_list_ = env_->GetMethodID(
        handler_cls, "getUserDefinedFunctionWhitelist", "()Ljava/lang/String;");
    if (!handler_get_udf_list_) {
      throw std::runtime_error(
          "cannot find CalciteServerHandler::getUserDefinedFunctionWhitelist method");
    }

    // Find 'CalciteServerHandler::getRuntimeExtensionFunctionWhitelist' method.
    handlhandler_get_rt_fn_list_ = env_->GetMethodID(
        handler_cls, "getRuntimeExtensionFunctionWhitelist", "()Ljava/lang/String;");
    if (!handlhandler_get_rt_fn_list_) {
      throw std::runtime_error(
          "cannot find CalciteServerHandler::getRuntimeExtensionFunctionWhitelist "
          "method");
    }

    // Find 'CalciteServerHandler::setRuntimeExtensionFunctions' method.
    handler_set_rt_fns_ = env_->GetMethodID(handler_cls,
                                            "setRuntimeExtensionFunctions",
                                            "(Ljava/util/List;Ljava/util/List;Z)V");
    if (!handler_set_rt_fns_) {
      throw std::runtime_error(
          "cannot find CalciteServerHandler::setRuntimeExtensionFunctions "
          "method");
    }
  }

  void findTQueryParsingOption() {
    parsing_opts_cls_ =
        env_->FindClass("com/omnisci/thrift/calciteserver/TQueryParsingOption");
    if (!parsing_opts_cls_) {
      throw std::runtime_error("cannot find Java class TQueryParsingOption");
    }
    parsing_opts_ctor_ = env_->GetMethodID(parsing_opts_cls_, "<init>", "(ZZZ)V");
    if (!parsing_opts_ctor_) {
      throw std::runtime_error("cannot find TQueryParsingOption ctor");
    }
  }

  void findTOptimizationOption() {
    optimization_opts_cls_ =
        env_->FindClass("com/omnisci/thrift/calciteserver/TOptimizationOption");
    if (!optimization_opts_cls_) {
      throw std::runtime_error("cannot find Java class TOptimizationOption");
    }
    optimization_opts_ctor_ =
        env_->GetMethodID(optimization_opts_cls_, "<init>", "(ZZLjava/util/List;)V");
    if (!optimization_opts_ctor_) {
      throw std::runtime_error("cannot find TOptimizationOption ctor");
    }
  }

  void findTPlanResult() {
    plan_result_cls_ = env_->FindClass("com/omnisci/thrift/calciteserver/TPlanResult");
    if (!plan_result_cls_) {
      throw std::runtime_error("cannot find Java class TPlanResult");
    }
    plan_result_plan_result_ =
        env_->GetFieldID(plan_result_cls_, "plan_result", "Ljava/lang/String;");
    if (!plan_result_plan_result_) {
      throw std::runtime_error("cannot find TPlanResult::plan_result field");
    }
  }

  void findTExtArgumentType() {
    jclass cls = env_->FindClass("com/omnisci/thrift/calciteserver/TExtArgumentType");
    if (!cls) {
      throw std::runtime_error("cannot find Java enum TExtArgumentType");
    }
    for (int i = 0; i < 41; ++i) {
      std::string val = to_string(static_cast<TExtArgumentType::type>(i));
      jfieldID field = env_->GetStaticFieldID(
          cls, val.c_str(), "Lcom/omnisci/thrift/calciteserver/TExtArgumentType;");
      if (!field) {
        throw std::runtime_error("cannot find Java enum TExtArgumentType::" + val +
                                 " field");
      }
      ext_arg_type_vals_.push_back(env_->GetStaticObjectField(cls, field));
    }
  }

  void findTOutputBufferSizeType() {
    jclass cls =
        env_->FindClass("com/omnisci/thrift/calciteserver/TOutputBufferSizeType");
    if (!cls) {
      throw std::runtime_error("cannot find Java enum TOutputBufferSizeType");
    }
    for (int i = 0; i < 4; ++i) {
      std::string val = to_string(static_cast<TOutputBufferSizeType::type>(i));
      jfieldID field = env_->GetStaticFieldID(
          cls, val.c_str(), "Lcom/omnisci/thrift/calciteserver/TOutputBufferSizeType;");
      if (!field) {
        throw std::runtime_error("cannot find Java enum TOutputBufferSizeType::" + val +
                                 " field");
      }
      output_buffer_size_type_vals_.push_back(env_->GetStaticObjectField(cls, field));
    }
  }

  void findTUserDefinedTableFunction() {
    udtfn_cls_ =
        env_->FindClass("com/omnisci/thrift/calciteserver/TUserDefinedTableFunction");
    if (!udtfn_cls_) {
      throw std::runtime_error("cannot find Java class TUserDefinedTableFunction");
    }
    udtfn_ctor_ = env_->GetMethodID(udtfn_cls_, "<init>", "()V");
    if (!udtfn_ctor_) {
      throw std::runtime_error("cannot find TUserDefinedTableFunction ctor");
    }
    // com.omnisci.thrift.calciteserver.TUserDefinedTableFunction
    // setName(java.lang.String);
    udtfn_set_name_ =
        env_->GetMethodID(udtfn_cls_,
                          "setName",
                          "(Ljava/lang/String;)Lcom/omnisci/thrift/calciteserver/"
                          "TUserDefinedTableFunction;");
    if (!udtfn_set_name_) {
      throw std::runtime_error("cannot find TUserDefinedTableFunction::setName method");
    }
    // com.omnisci.thrift.calciteserver.TUserDefinedTableFunction
    // setSizerType(com.omnisci.thrift.calciteserver.TOutputBufferSizeType);
    udtfn_set_sizer_type_ = env_->GetMethodID(
        udtfn_cls_,
        "setSizerType",
        "(Lcom/omnisci/thrift/calciteserver/TOutputBufferSizeType;)Lcom/omnisci/thrift/"
        "calciteserver/TUserDefinedTableFunction;");
    if (!udtfn_set_sizer_type_) {
      throw std::runtime_error(
          "cannot find TUserDefinedTableFunction::setSizerType method");
    }
    // com.omnisci.thrift.calciteserver.TUserDefinedTableFunction setSizerArgPos(int);
    udtfn_set_sizer_arg_pos_ = env_->GetMethodID(
        udtfn_cls_,
        "setSizerArgPos",
        "(I)Lcom/omnisci/thrift/calciteserver/TUserDefinedTableFunction;");
    if (!udtfn_set_sizer_arg_pos_) {
      throw std::runtime_error(
          "cannot find TUserDefinedTableFunction::setSizerArgPos method");
    }
    // com.omnisci.thrift.calciteserver.TUserDefinedTableFunction
    // setInputArgTypes(java.util.List<com.omnisci.thrift.calciteserver.TExtArgumentType>);
    udtfn_set_input_arg_types_ = env_->GetMethodID(
        udtfn_cls_,
        "setInputArgTypes",
        "(Ljava/util/List;)Lcom/omnisci/thrift/calciteserver/TUserDefinedTableFunction;");
    if (!udtfn_set_input_arg_types_) {
      throw std::runtime_error(
          "cannot find TUserDefinedTableFunction::setInputArgTypes method");
    }
    // com.omnisci.thrift.calciteserver.TUserDefinedTableFunction
    // setOutputArgTypes(java.util.List<com.omnisci.thrift.calciteserver.TExtArgumentType>);
    udtfn_set_output_arg_types_ = env_->GetMethodID(
        udtfn_cls_,
        "setOutputArgTypes",
        "(Ljava/util/List;)Lcom/omnisci/thrift/calciteserver/TUserDefinedTableFunction;");
    if (!udtfn_set_output_arg_types_) {
      throw std::runtime_error(
          "cannot find TUserDefinedTableFunction::setOutputArgTypes method");
    }
    // com.omnisci.thrift.calciteserver.TUserDefinedTableFunction
    // setSqlArgTypes(java.util.List<com.omnisci.thrift.calciteserver.TExtArgumentType>);
    udtfn_set_sql_arg_types_ = env_->GetMethodID(
        udtfn_cls_,
        "setSqlArgTypes",
        "(Ljava/util/List;)Lcom/omnisci/thrift/calciteserver/TUserDefinedTableFunction;");
    if (!udtfn_set_sql_arg_types_) {
      throw std::runtime_error(
          "cannot find TUserDefinedTableFunction::setSqlArgTypes method");
    }
    // com.omnisci.thrift.calciteserver.TUserDefinedTableFunction
    // setAnnotations(java.util.List<java.util.Map<java.lang.String, java.lang.String>>);
    udtfn_set_annotations_ = env_->GetMethodID(
        udtfn_cls_,
        "setAnnotations",
        "(Ljava/util/List;)Lcom/omnisci/thrift/calciteserver/TUserDefinedTableFunction;");
    if (!udtfn_set_annotations_) {
      throw std::runtime_error(
          "cannot find TUserDefinedTableFunction::setAnnotations method");
    }
  }

  void findInvalidParseRequest() {
    invalid_parse_req_cls_ =
        env_->FindClass("com/omnisci/thrift/calciteserver/InvalidParseRequest");
    if (!invalid_parse_req_cls_) {
      throw std::runtime_error("cannot find Java class InvalidParseRequest");
    }
    invalid_parse_req_why_ =
        env_->GetFieldID(invalid_parse_req_cls_, "whyUp", "Ljava/lang/String;");
    if (!invalid_parse_req_why_) {
      throw std::runtime_error("cannot find InvalidParseRequest::whyUp field");
    }
  }

  void findArrayList() {
    array_list_cls_ = env_->FindClass("java/util/ArrayList");
    if (!array_list_cls_) {
      throw std::runtime_error("cannot find Java class ArrayList");
    }
    array_list_ctor_ = env_->GetMethodID(array_list_cls_, "<init>", "()V");
    if (!array_list_ctor_) {
      throw std::runtime_error("cannot find ArrayList ctor");
    }
    array_list_add_ = env_->GetMethodID(array_list_cls_, "add", "(Ljava/lang/Object;)Z");
    if (!array_list_add_) {
      throw std::runtime_error("cannot find ArrayList::add method");
    }
  }

  void findHashMap() {
    hash_map_cls_ = env_->FindClass("java/util/HashMap");
    if (!hash_map_cls_) {
      throw std::runtime_error("cannot find Java class HashMap");
    }
    hash_map_ctor_ = env_->GetMethodID(hash_map_cls_, "<init>", "()V");
    if (!hash_map_ctor_) {
      throw std::runtime_error("cannot find HashMap ctor");
    }
    hash_map_put_ = env_->GetMethodID(
        hash_map_cls_, "put", "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");
    if (!hash_map_put_) {
      throw std::runtime_error("cannot find HashMap::put method");
    }
  }

  std::string convertJavaString(jstring str_obj) {
    const char* res_str = env_->GetStringUTFChars(str_obj, 0);
    std::string res = res_str;
    env_->ReleaseStringUTFChars(str_obj, res_str);
    return res;
  }

  std::string readStringField(jobject obj, jfieldID field) {
    auto field_obj = (jstring)env_->GetObjectField(obj, field);
    return convertJavaString(field_obj);
  }

  template <typename T>
  jobject convertEnumValue(T val, const std::vector<jobject>& enum_vals) {
    size_t index = static_cast<size_t>(val);
    if (index < enum_vals.size()) {
      return enum_vals[index];
    }
    std::runtime_error("Unsupported enum value: " + to_string(val));
  }

  jobject convertTOutputBufferSizeType(TOutputBufferSizeType::type val) {
    return convertEnumValue(val, output_buffer_size_type_vals_);
  }

  jobject convertTExtArgumentType(TExtArgumentType::type val) {
    return convertEnumValue(val, ext_arg_type_vals_);
  }

  jobject convertTExtArgumentTypes(const std::vector<TExtArgumentType::type>& vals) {
    jobject res = env_->NewObject(array_list_cls_, array_list_ctor_);
    for (auto& val : vals) {
      env_->CallVoidMethod(res, array_list_add_, convertTExtArgumentType(val));
    }
    return res;
  }

  jobject convertTUserDefinedTableFunction(const TUserDefinedTableFunction& udtf) {
    jobject res = env_->NewObject(udtfn_cls_, udtfn_ctor_);
    // java.lang.String name;
    env_->CallObjectMethod(res, udtfn_set_name_, env_->NewStringUTF(udtf.name.c_str()));
    // TOutputBufferSizeType sizerType;
    env_->CallObjectMethod(
        res, udtfn_set_sizer_type_, convertTOutputBufferSizeType(udtf.sizerType));
    // int sizerArgPos;
    env_->CallObjectMethod(res, udtfn_set_sizer_arg_pos_, (jint)udtf.sizerArgPos);
    // java.util.List<TExtArgumentType> inputArgTypes;
    env_->CallVoidMethod(
        res, udtfn_set_input_arg_types_, convertTExtArgumentTypes(udtf.inputArgTypes));
    // java.util.List<TExtArgumentType> outputArgTypes;
    env_->CallVoidMethod(
        res, udtfn_set_output_arg_types_, convertTExtArgumentTypes(udtf.outputArgTypes));
    // java.util.List<TExtArgumentType> sqlArgTypes;
    env_->CallVoidMethod(
        res, udtfn_set_sql_arg_types_, convertTExtArgumentTypes(udtf.sqlArgTypes));
    // java.util.List<java.util.Map<java.lang.String,java.lang.String>> annotations;
    jobject annotations = env_->NewObject(array_list_cls_, array_list_ctor_);
    for (auto& ann : udtf.annotations) {
      jobject jmap = env_->NewObject(hash_map_cls_, hash_map_ctor_);
      for (auto& [key, val] : ann) {
        auto jkey = env_->NewStringUTF(key.c_str());
        auto jval = env_->NewStringUTF(val.c_str());
        env_->CallVoidMethod(jmap, hash_map_put_, jkey, jval);
      }
      env_->CallVoidMethod(annotations, array_list_add_, jmap);
    }
    env_->CallVoidMethod(res, udtfn_set_annotations_, annotations);
    return res;
  }

  // Java machine and environment.
  JavaVM* jvm_;
  JNIEnv* env_;

  // com.mapd.parser.server.CalciteServerHandler instance and methods.
  jobject handler_obj_;
  jmethodID handler_process_;
  jmethodID handler_get_ext_fn_list_;
  jmethodID handler_get_udf_list_;
  jmethodID handlhandler_get_rt_fn_list_;
  jmethodID handler_set_rt_fns_;

  // com.omnisci.thrift.calciteserver.TQueryParsingOption class and methods
  jclass parsing_opts_cls_;
  jmethodID parsing_opts_ctor_;

  // com.omnisci.thrift.calciteserver.TOptimizationOption class and methods
  jclass optimization_opts_cls_;
  jmethodID optimization_opts_ctor_;

  // com.omnisci.thrift.calciteserver.TPlanResult class and fields
  jclass plan_result_cls_;
  jfieldID plan_result_plan_result_;

  // com.omnisci.thrift.calciteserver.TExtArgumentType enum values
  std::vector<jobject> ext_arg_type_vals_;

  // com.omnisci.thrift.calciteserver.TOutputBufferSizeType enum values
  std::vector<jobject> output_buffer_size_type_vals_;

  // com.omnisci.thrift.calciteserver.TUserDefinedTableFunction class and methods
  jclass udtfn_cls_;
  jmethodID udtfn_ctor_;
  jmethodID udtfn_set_name_;
  jmethodID udtfn_set_sizer_type_;
  jmethodID udtfn_set_sizer_arg_pos_;
  jmethodID udtfn_set_input_arg_types_;
  jmethodID udtfn_set_output_arg_types_;
  jmethodID udtfn_set_sql_arg_types_;
  jmethodID udtfn_set_annotations_;

  // com.omnisci.thrift.calciteserver.InvalidParseRequest class and fields
  jclass invalid_parse_req_cls_;
  jfieldID invalid_parse_req_why_;

  // java.util.ArrayList class and methods
  jclass array_list_cls_;
  jmethodID array_list_ctor_;
  jmethodID array_list_add_;

  // java.util.HashMap class and methods
  jclass hash_map_cls_;
  jmethodID hash_map_ctor_;
  jmethodID hash_map_put_;
};

CalciteJNI::CalciteJNI(const std::string& udf_filename, size_t calcite_max_mem_mb) {
  impl_ = std::make_unique<Impl>(udf_filename, calcite_max_mem_mb);
}

CalciteJNI::~CalciteJNI() {}

TPlanResult CalciteJNI::process(
    const std::string& user,
    const std::string& db_name,
    const std::string& sql_string,
    const std::string& schema_json,
    const std::string& session_id,
    const std::vector<TFilterPushDownInfo>& filter_push_down_info,
    const bool legacy_syntax,
    const bool is_explain,
    const bool is_view_optimize) {
  return impl_->process(user,
                        db_name,
                        sql_string,
                        schema_json,
                        session_id,
                        filter_push_down_info,
                        legacy_syntax,
                        is_explain,
                        is_view_optimize);
}

std::string CalciteJNI::getExtensionFunctionWhitelist() {
  return impl_->getExtensionFunctionWhitelist();
}
std::string CalciteJNI::getUserDefinedFunctionWhitelist() {
  return impl_->getUserDefinedFunctionWhitelist();
}
std::string CalciteJNI::getRuntimeExtensionFunctionWhitelist() {
  return impl_->getRuntimeExtensionFunctionWhitelist();
}
void CalciteJNI::setRuntimeExtensionFunctions(
    const std::vector<TUserDefinedFunction>& udfs,
    const std::vector<TUserDefinedTableFunction>& udtfs,
    bool is_runtime) {
  return impl_->setRuntimeExtensionFunctions(udfs, udtfs, is_runtime);
}
