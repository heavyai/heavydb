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
  Calcite(int port, std::string data_dir);
  Calcite();
  std::string process(std::string user,
                      std::string passwd,
                      std::string catalog,
                      std::string sql_string,
                      const bool legacy_syntax);
  virtual ~Calcite();

 private:
  std::unique_ptr<CalciteServerClient> client;
  bool server_available_;
  bool jni_;
  JavaVM* jvm_;
  jclass calciteDirect_;
  jobject calciteDirectObject_;
  jmethodID constructor_;
  jmethodID processMID_;
  jmethodID hasFailedMID_;
  jmethodID getElapsedTimeMID_;
  jmethodID getTextMID_;
};

#endif /* CALCITE_H */
