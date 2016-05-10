/*
 *  Some cool MapD License
 */

/*
 * File:   Calcite.cpp
 * Author: michael
 *
 * Created on November 23, 2015, 9:33 AM
 */

#include "Calcite.h"
#include "Shared/measure.h"
#include "../Shared/mapdpath.h"

#include <glog/logging.h>

#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/transport/TSocket.h>
#include <thrift/transport/TTransportUtils.h>

#include "gen-cpp/CalciteServer.h"

using namespace std;
using namespace apache::thrift;
using namespace apache::thrift::protocol;
using namespace apache::thrift::transport;

void Calcite::runJNI(int port, std::string data_dir) {
  LOG(INFO) << "Creating Calcite Server local as JNI instance";
  const int kNumOptions = 3;
  std::string jar_file{"-Djava.class.path=" + mapd_root_abs_path() +
                       "/bin/mapd-1.0-SNAPSHOT-jar-with-dependencies.jar"};
  JavaVMOption options[kNumOptions] = {{const_cast<char*>("-Xmx256m"), NULL},
                                       {const_cast<char*>("-verbose:gc"), NULL},
                                       {const_cast<char*>(jar_file.c_str()), NULL}};

  JavaVMInitArgs vm_args;
  vm_args.version = JNI_VERSION_1_6;
  vm_args.options = options;
  vm_args.nOptions = sizeof(options) / sizeof(JavaVMOption);
  assert(vm_args.nOptions == kNumOptions);

  JNIEnv* env;
  int res = JNI_CreateJavaVM(&jvm_, reinterpret_cast<void**>(&env), &vm_args);
  CHECK_EQ(res, JNI_OK);

  calciteDirect_ = env->FindClass("com/mapd/parser/server/CalciteDirect");

  CHECK(calciteDirect_);

  // now call the constructor
  constructor_ = env->GetMethodID(calciteDirect_, "<init>", "(ILjava/lang/String;)V");
  CHECK(constructor_);

  // create the new calciteDirect via call to constructor
  calciteDirectObject_ =
      env->NewGlobalRef(env->NewObject(calciteDirect_, constructor_, port, env->NewStringUTF(data_dir.c_str())));
  CHECK(calciteDirectObject_);

  // get all the methods we will need for calciteDirect;
  processMID_ = env->GetMethodID(calciteDirect_,
                                 "process",
                                 "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Z)Lcom/"
                                 "mapd/parser/server/CalciteReturn;");
  CHECK(processMID_);

  // get all the methods we will need for calciteDirect;
  updateMetadataMID_ = env->GetMethodID(calciteDirect_, "updateMetadata", "(Ljava/lang/String;)V");
  CHECK(updateMetadataMID_);

  // get all the methods we will need to process the calcite results
  jclass calcite_return_class = env->FindClass("com/mapd/parser/server/CalciteReturn");
  CHECK(calcite_return_class);

  hasFailedMID_ = env->GetMethodID(calcite_return_class, "hasFailed", "()Z");
  CHECK(hasFailedMID_);
  getElapsedTimeMID_ = env->GetMethodID(calcite_return_class, "getElapsedTime", "()J");
  CHECK(getElapsedTimeMID_);
  getTextMID_ = env->GetMethodID(calcite_return_class, "getText", "()Ljava/lang/String;");
  CHECK(getTextMID_);
}

void Calcite::runServer(int port, std::string data_dir) {
  LOG(INFO) << "Remote calcite server";
  server_available_ = true;
  jni_ = false;
  // check server is responding
  boost::shared_ptr<TTransport> socket(new TSocket("localhost", port));
  boost::shared_ptr<TTransport> transport(new TBufferedTransport(socket));
  boost::shared_ptr<TProtocol> protocol(new TBinaryProtocol(transport));
  client.reset(new CalciteServerClient(protocol));

  try {
    transport->open();

    auto ms = measure<>::execution([&]() { client->ping(); });

    LOG(INFO) << "ping took " << ms << " ms " << endl;

  } catch (TException& tx) {
    LOG(ERROR) << tx.what() << endl;
    server_available_ = false;
    LOG(ERROR) << "No calcite remote server running on port " << port;
  }
}

Calcite::Calcite(int port, std::string data_dir) : server_available_(false), jni_(true), jvm_(NULL) {
  LOG(INFO) << "Creating Calcite Handler,  Calcite Port is " << port << " base data dir is " << data_dir;
  if (port == -1) {
    runJNI(port, data_dir);
  } else {
    remote_calcite_port_ = port;
    runServer(port, data_dir);
  }
}

JNIEnv* Calcite::checkJNIConnection() {
  JNIEnv* env;
  int res = jvm_->GetEnv((void**)&env, JNI_VERSION_1_6);
  if (res != JNI_OK) {
    JavaVMAttachArgs args;
    args.version = JNI_VERSION_1_6;  // choose your JNI version
    args.name = NULL;                // you might want to give the java thread a name
    args.group = NULL;               // you might want to assign the java thread to a ThreadGroup
    int res = jvm_->AttachCurrentThread((void**)&env, &args);
    CHECK_EQ(res, JNI_OK);
  }
  CHECK(calciteDirectObject_);
  CHECK(processMID_);
  CHECK(updateMetadataMID_);

  return env;
}

void Calcite::updateMetadata(string metadata) {
  if (jni_) {
    JNIEnv* env = checkJNIConnection();
    auto ms = measure<>::execution([&]() {
      env->CallObjectMethod(calciteDirectObject_, updateMetadataMID_, env->NewStringUTF(metadata.c_str()));

    });
    if (env->ExceptionCheck()) {
      LOG(ERROR) << "Exception occured ";
      env->ExceptionDescribe();
      LOG(ERROR) << "Exception occured " << env->ExceptionOccurred();
      throw std::runtime_error("Calcite::updateMetadata failed");
    }
    LOG(INFO) << "Time to updateMetadata " << ms << " (ms)" << endl;
    jvm_->DetachCurrentThread();
  }
}

string Calcite::process(string user, string passwd, string catalog, string sql_string, const bool legacy_syntax) {
  LOG(INFO) << "User " << user << " catalog " << catalog << " sql '" << sql_string << "'";
  if (jni_) {
    JNIEnv* env = checkJNIConnection();
    jboolean legacy = legacy_syntax;
    jobject process_result;
    auto ms = measure<>::execution([&]() {
      process_result = env->CallObjectMethod(calciteDirectObject_,
                                             processMID_,
                                             env->NewStringUTF(user.c_str()),
                                             env->NewStringUTF(passwd.c_str()),
                                             env->NewStringUTF(catalog.c_str()),
                                             env->NewStringUTF(sql_string.c_str()),
                                             legacy);

    });
    if (env->ExceptionCheck()) {
      LOG(ERROR) << "Exception occured ";
      env->ExceptionDescribe();
      LOG(ERROR) << "Exception occured " << env->ExceptionOccurred();
      throw std::runtime_error("Calcite::process failed");
    }
    CHECK(process_result);
    long java_time = env->CallLongMethod(process_result, getElapsedTimeMID_);

    LOG(INFO) << "Time marshalling in JNI " << (ms > java_time ? ms - java_time : 0) << " (ms), Time in Java Calcite  "
              << java_time << " (ms)" << endl;
    CHECK(process_result);
    CHECK(getTextMID_);
    jstring s = (jstring)env->CallObjectMethod(process_result, getTextMID_);
    CHECK(s);
    // convert the Java String to use it in C
    jboolean iscopy;
    const char* text = env->GetStringUTFChars(s, &iscopy);

    bool failed = env->CallBooleanMethod(process_result, hasFailedMID_);
    if (failed) {
      LOG(ERROR) << "Calcite process failed " << text;
      string retText(text);
      env->ReleaseStringUTFChars(s, text);
      env->DeleteLocalRef(process_result);
      jvm_->DetachCurrentThread();
      throw std::invalid_argument(retText);
    }
    string retText(text);
    env->ReleaseStringUTFChars(s, text);
    env->DeleteLocalRef(process_result);
    jvm_->DetachCurrentThread();
    return retText;
  } else {
    if (server_available_) {
      TPlanResult ret;
      auto ms = measure<>::execution([&]() { client->process(ret, user, passwd, catalog, sql_string, legacy_syntax); });
      LOG(INFO) << ret.plan_result << endl;
      LOG(INFO) << "Time in Thrift " << (ms > ret.execution_time_ms ? ms - ret.execution_time_ms : 0)
                << " (ms), Time in Java Calcite server " << ret.execution_time_ms << " (ms)" << endl;
      return ret.plan_result;
    } else {
      LOG(INFO) << "Not routing to Calcite, server is not up and JNI not available" << endl;
      return "";
    }
  }
}

Calcite::~Calcite() {
  LOG(INFO) << "Destroy Calcite Class" << std::endl;
  if (jvm_) {
    jvm_->DestroyJavaVM();
  }
  LOG(INFO) << "After java destroy: ";
}