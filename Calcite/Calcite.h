/*
 *  Some cool MapD License
 */

/*
 * File:   Calcite.h
 * Author: michael
 *
 * Created on November 23, 2015, 9:33 AM
 */

#ifndef CALCITE_H
#define CALCITE_H

#include "gen-cpp/CalciteServer.h"
#include <jni.h>

class Calcite {
 public:
  Calcite(int port, std::string data_dir, size_t calcite_max_mem);
  std::string process(std::string user,
                      std::string passwd,
                      std::string catalog,
                      std::string sql_string,
                      const bool legacy_syntax,
                      const bool is_explain);
  std::string getExtensionFunctionWhitelist();
  void updateMetadata(std::string catalog, std::string table);
  virtual ~Calcite();

 private:
  void runJNI(int port, std::string data_dir, size_t calcite_max_memory);
  void runServer(int port, std::string data_dir);
  std::string handle_java_return(JNIEnv* env, jobject process_result);
  JNIEnv* checkJNIConnection();
  std::unique_ptr<CalciteServerClient> client;
  bool server_available_;
  bool jni_;
  int remote_calcite_port_ = -1;
  JavaVM* jvm_;
  jclass calciteDirect_;
  jobject calciteDirectObject_;
  jmethodID constructor_;
  jmethodID processMID_;
  jmethodID updateMetadataMID_;
  jmethodID hasFailedMID_;
  jmethodID getElapsedTimeMID_;
  jmethodID getTextMID_;
  jmethodID getExtensionFunctionWhitelistMID_;
};

#endif /* CALCITE_H */
