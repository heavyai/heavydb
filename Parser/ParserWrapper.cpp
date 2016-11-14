/*
 *  Some cool MapD License
 */

/*
 * File:   ParserWrapper.cpp
 * Author: michael
 *
 * Created on Feb 23, 2016, 9:33 AM
 */

#include "ParserWrapper.h"
#include "Shared/measure.h"

#include <boost/algorithm/string.hpp>

#include <glog/logging.h>

using namespace std;

const std::vector<std::string> ParserWrapper::ddl_cmd = {"ALTER", "COPY", "GRANT", "CREATE", "DROP"};

const std::vector<std::string> ParserWrapper::update_dml_cmd = {
    "INSERT",
    "DELETE",
    "UPDATE",
    "UPSERT",
};

const std::string ParserWrapper::explain_str = {"explain"};
const std::string ParserWrapper::calcite_explain_str = {"explain calcite"};

ParserWrapper::ParserWrapper(std::string query_string) {
  if (boost::istarts_with(query_string, calcite_explain_str)) {
    actual_query = boost::trim_copy(query_string.substr(calcite_explain_str.size()));
    ParserWrapper inner{actual_query};
    if (inner.is_ddl || inner.is_update_dml) {
      is_other_explain = true;
      return;
    } else {
      is_select_calcite_explain = true;
      return;
    }
  }

  if (boost::istarts_with(query_string, explain_str)) {
    actual_query = boost::trim_copy(query_string.substr(explain_str.size()));
    ParserWrapper inner{actual_query};
    if (inner.is_ddl || inner.is_update_dml) {
      is_other_explain = true;
      return;
    } else {
      is_select_explain = true;
      return;
    }
  }

  for (std::string ddl : ddl_cmd) {
    is_ddl = boost::istarts_with(query_string, ddl);
    if (is_ddl) {
      return;
    }
  }

  for (std::string u_dml : update_dml_cmd) {
    is_update_dml = boost::istarts_with(query_string, u_dml);
    if (is_update_dml) {
      return;
    }
  }
}

ParserWrapper::~ParserWrapper() {
}
