#ifndef REGEX_SUPPORT_H
#define REGEX_SUPPORT_H

#include "boost/regex.hpp"

template <typename COMMAND_LIST_TYPE, typename CONTEXT_TYPE, typename REGEX_PRESENT_LAMBDA>
bool on_valid_regex_present(COMMAND_LIST_TYPE const& command_list,
                            CONTEXT_TYPE& client_context,
                            REGEX_PRESENT_LAMBDA on_present_op,
                            std::ostream& output_stream = std::cout) {
  try {
    if (command_list.size() > 1) {
      boost::regex filter_expression(command_list[1], boost::regex_constants::no_except);
      if (filter_expression.status() == 0) {  // On successful regex
        on_present_op(filter_expression, client_context);
        return true;
      } else {
        output_stream << "Malformed regular expression: " << command_list[1] << std::endl;
      }
    }
  } catch (
      std::runtime_error& re) {  // Handles cases outlined in boost regex bad expressions test case, to avoid blow ups
    output_stream << "Invalid regular expression: " << command_list[1] << std::endl;
  }

  return false;
}

std::function<bool(std::string const&)> yield_default_filter_function(boost::regex& filter_expression) {
  return [&filter_expression](std::string const& input) -> bool {
    boost::smatch results;
    return !boost::regex_match(input, results, filter_expression);
  };
}

template <ThriftService SERVICE_TYPE, typename CLIENT_CONTEXT_TYPE, typename DETAIL_PROCESSOR>
void for_all_return_names(CLIENT_CONTEXT_TYPE& context, DETAIL_PROCESSOR detail_processor) {
  thrift_op<SERVICE_TYPE>(context, [&](CLIENT_CONTEXT_TYPE& lambda_context) {
    for (auto name_val : lambda_context.names_return) {
      detail_processor(name_val, lambda_context);
    }
  });
}

template <ThriftService THRIFT_SERVICE, typename COMMAND_LIST_TYPE, typename CONTEXT_TYPE>
void returned_list_regex(COMMAND_LIST_TYPE const& command_list,
                         CONTEXT_TYPE& context,
                         std::ostream& output_stream = std::cout) {
  // clang-format off
  auto did_execute = on_valid_regex_present(command_list, context, [&output_stream](boost::regex& filter_expression, CONTEXT_TYPE& context_param) {
    auto filter_function = yield_default_filter_function(filter_expression);
    auto result_processor = [filter_function, &output_stream](std::string const& element_name, CONTEXT_TYPE& context_param) {
      if (!filter_function(element_name)) {
        output_stream << element_name << '\n';
      }
    };

    for_all_return_names<THRIFT_SERVICE>(context_param, result_processor);
  });
  // clang-format on

  if (!did_execute) {  // Run classic mode instead
    for_all_return_names<THRIFT_SERVICE>(
        context, [&output_stream](std::string const& element_name, CONTEXT_TYPE& context_param) {
          output_stream << element_name << '\n';
        });
  }

  output_stream.flush();
}

#endif
