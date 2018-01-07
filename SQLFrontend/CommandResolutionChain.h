#ifndef COMMAND_RESOLUTION_CHAIN_H
#define COMMAND_RESOLUTION_CHAIN_H

#include "gtest/gtest.h"

#include <sstream>
#include <vector>

template <typename TOKEN_LIST_TYPE>
class DefaultInputParser {
 public:
  template <typename INPUT_TYPE>
  TOKEN_LIST_TYPE extract_tokens(INPUT_TYPE& input_stream) {
    TOKEN_LIST_TYPE token_list;

    std::istringstream input_resolver(input_stream);
    std::string input_token;
    while (input_resolver >> input_token) {
      token_list.push_back(input_token);
    }

    return token_list;
  }
};

template <template <typename> class INPUT_PARSER_TYPE = DefaultInputParser>
class CommandResolutionChain {
 public:
  FRIEND_TEST(MapDQLTest, CommandResolutionChain_DefaultTokenizer);

  using CommandTokenList = std::vector<std::string>;
  using ParamCountType = CommandTokenList::size_type;
  using TokenExtractor = INPUT_PARSER_TYPE<CommandTokenList>;

  CommandResolutionChain(CommandResolutionChain const&) = delete;
  CommandResolutionChain() = delete;
  CommandResolutionChain& operator=(CommandResolutionChain const&) = delete;

  template <typename RESOLUTION_FUNCTOR>
  CommandResolutionChain(char const* command_line,
                         std::string const& target_command,
                         ParamCountType min_param_count,
                         RESOLUTION_FUNCTOR resolution_op,
                         std::string const& custom_help)
      : m_resolved(false) {
    m_command_token_list = TokenExtractor().extract_tokens(command_line);

    resolve_command(target_command,
                    min_param_count,
                    resolution_op,
                    custom_help_string(target_command, min_param_count, custom_help));
  }

  template <typename RESOLUTION_FUNCTOR>
  CommandResolutionChain(char const* command_line,
                         std::string const& target_command,
                         ParamCountType min_param_count,
                         RESOLUTION_FUNCTOR resolution_op)
      : m_resolved(false) {
    m_command_token_list = TokenExtractor().extract_tokens(command_line);

    resolve_command(
        target_command, min_param_count, resolution_op, default_help_string(target_command, min_param_count));
  }

  template <typename RESOLUTION_FUNCTOR>
  CommandResolutionChain& operator()(std::string const& target_command,
                                     ParamCountType min_param_count,
                                     RESOLUTION_FUNCTOR resolution_op,
                                     std::string const& custom_help) {
    if (m_resolved == true)
      return (*this);
    resolve_command(target_command,
                    min_param_count,
                    resolution_op,
                    custom_help_string(target_command, min_param_count, custom_help));
    return (*this);
  }

  template <typename RESOLUTION_FUNCTOR>
  CommandResolutionChain& operator()(std::string const& target_command,
                                     ParamCountType min_param_count,
                                     RESOLUTION_FUNCTOR resolution_op) {
    if (m_resolved == true)
      return (*this);
    resolve_command(
        target_command, min_param_count, resolution_op, default_help_string(target_command, min_param_count));
    return (*this);
  }

  bool is_resolved() { return m_resolved; }

 private:
  std::string custom_help_string(std::string const& target_command,
                                 ParamCountType min_param_count,
                                 std::string const& custom_help) {
    std::ostringstream help_stream;
    help_stream << target_command << " insufficient parameters provided; need " << min_param_count - 1
                << " total parameter(s)\n"
                << custom_help << std::endl;
    return help_stream.str();
  }

  std::string default_help_string(std::string const& target_command, ParamCountType min_param_count) {
    std::ostringstream help_stream;
    help_stream << target_command << " insufficient parameters provided; need " << min_param_count - 1
                << " total parameter(s)" << std::endl;
    return help_stream.str();
  }

  template <typename RESOLUTION_FUNCTOR>
  void resolve_command(std::string const& target_command,
                       ParamCountType min_param_count,
                       RESOLUTION_FUNCTOR resolution_op,
                       std::string const& help_info) {
    if (m_command_token_list.empty()) {
      m_resolved = true;
    } else if (m_command_token_list[0] == target_command) {
      if (m_command_token_list.size() < min_param_count) {
        std::cout << help_info << '\n';
      } else {
        resolution_op(m_command_token_list);
      }
      m_resolved = true;
      ;
    }
  }

  CommandTokenList m_command_token_list;
  bool m_resolved;
};

#endif
