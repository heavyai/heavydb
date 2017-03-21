/**
 * @file    ParserWrapper.h
 * @author  michael
 * @brief   Classes used to wrap parser calls for calcite redirection
 *
 * Copyright (c) 2016 MapD Technologies, Inc.  All rights reserved.
 **/
#ifndef PARSER_WRAPPER_H_
#define PARSER_WRAPPER_H_

#include <string>
#include <vector>

class ParserWrapper {
 public:
  ParserWrapper(std::string query_string);
  std::string process(std::string user,
                      std::string passwd,
                      std::string catalog,
                      std::string sql_string,
                      const bool legacy_syntax);
  virtual ~ParserWrapper();
  bool is_select_explain = false;
  bool is_select_calcite_explain = false;
  bool is_other_explain = false;
  bool is_ddl = false;
  bool is_update_dml = false;
  bool is_copy = false;
  std::string actual_query;

 private:
  static const std::vector<std::string> ddl_cmd;
  static const std::vector<std::string> update_dml_cmd;
  static const std::string explain_str;
  static const std::string calcite_explain_str;
};

#endif  // PARSERWRAPPER_H_
