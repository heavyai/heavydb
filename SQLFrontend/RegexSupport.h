#ifndef REGEX_SUPPORT_H
#define REGEX_SUPPORT_H

#include "boost/regex.hpp"

template <typename COMMAND_TYPE, typename CONTEXT_TYPE, typename REGEX_PRESENT_LAMBDA>
bool on_valid_regex_present(COMMAND_TYPE const& command_input,
                            CONTEXT_TYPE& client_context,
                            REGEX_PRESENT_LAMBDA on_present_op) {
  std::string cmd_extraction, regex_extraction;
  std::istringstream input_tokens(command_input);
  input_tokens >> cmd_extraction >> regex_extraction;

  try {
    if (regex_extraction.size() > 0) {
      boost::regex filter_expression(regex_extraction, boost::regex_constants::no_except);
      if (filter_expression.status() == 0) {  // On successful regex
        on_present_op(filter_expression, client_context);
        return true;
      } else {
        std::cout << "Malformed regular expression: " << regex_extraction << std::endl;
      }
    }
  } catch (
      std::runtime_error& re) {  // Handles cases outlined in boost regex bad expressions test case, to avoid blow ups
    std::cout << "Invalid regular expression: " << regex_extraction << std::endl;
  }

  return false;
}

std::function<bool(std::string const&)> yield_default_filter_function(boost::regex& filter_expression) {
  return [&filter_expression](std::string const& input) -> bool {
    boost::smatch results;
    return !boost::regex_match(input, results, filter_expression);
  };
}

#endif
