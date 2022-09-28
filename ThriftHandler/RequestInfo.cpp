#include "RequestInfo.h"

#include <boost/property_tree/json_parser.hpp>

#include <sstream>

namespace heavyai {

RequestInfo::RequestInfo(std::string const& session_id_or_json) {
  if (!session_id_or_json.empty() && session_id_or_json[0] == '{') {
    std::istringstream iss(session_id_or_json);
    boost::property_tree::ptree ptree;
    boost::property_tree::read_json(iss, ptree);
    session_id_ = ptree.get<std::string>("session_id");
    request_id_ = ptree.get<logger::RequestId>("request_id");
  } else {
    session_id_ = session_id_or_json;
    request_id_ = 0;  // Valid request_ids are always positive.
  }
}

std::string RequestInfo::json() const {
  boost::property_tree::ptree ptree;
  std::ostringstream oss;
  ptree.put("session_id", session_id_);
  ptree.put("request_id", request_id_);
  constexpr bool pretty_print = false;
  boost::property_tree::write_json(oss, ptree, pretty_print);
  return oss.str();
}

}  // namespace heavyai
