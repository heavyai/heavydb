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

class Calcite {
 public:
  Calcite(int port);
  std::string process(std::string user, std::string passwd, std::string catalog, std::string sql_string);
  void init(int port);
  virtual ~Calcite();

 private:
  std::unique_ptr<CalciteServerClient> client;
  bool server_available;
};

#endif /* CALCITE_H */
