#ifndef COMMAND_RESOLUTION_CHAIN_H
#define COMMAND_RESOLUTION_CHAIN_H

#include "CommandDeterminant.h"
#include "InputTokenizers.h"
#include "gtest/gtest.h"

#include <sstream>
#include <vector>

template <template <typename> class INPUT_PARSER_TYPE = QuotedInputSupportParser,
          template <typename> class REGEX_INPUT_PARSER_TYPE = DefaultInputParser>
class CommandResolutionChain {
 public:
  FRIEND_TEST(OmniSQLTest, CommandResolutionChain_DefaultTokenizer);

  using CommandTokenList = std::vector<std::string>;
  using ParamCountType = CommandTokenList::size_type;
  using StandardTokenExtractor = INPUT_PARSER_TYPE<CommandTokenList>;
  using RegexTokenExtractor = REGEX_INPUT_PARSER_TYPE<CommandTokenList>;

  CommandResolutionChain(CommandResolutionChain const&) = delete;
  CommandResolutionChain() = delete;
  CommandResolutionChain& operator=(CommandResolutionChain const&) = delete;

  template <typename RESOLUTION_FUNCTOR>
  CommandResolutionChain(char const* command_line,
                         std::string const& target_command,
                         ParamCountType min_param_count,
                         ParamCountType max_param_count,
                         RESOLUTION_FUNCTOR resolution_op,
                         std::string const& custom_help,
                         std::ostream& output_stream = std::cout)
      : m_resolved(false), m_output_stream(output_stream) {
    extractTokens(command_line);

    resolve_command(target_command,
                    min_param_count,
                    max_param_count,
                    resolution_op,
                    custom_help_string(
                        target_command, min_param_count, max_param_count, custom_help));
  }

  template <typename RESOLUTION_FUNCTOR>
  CommandResolutionChain(char const* command_line,
                         std::string const& target_command,
                         ParamCountType min_param_count,
                         ParamCountType max_param_count,
                         RESOLUTION_FUNCTOR resolution_op,
                         std::ostream& output_stream = std::cout)
      : m_resolved(false), m_output_stream(output_stream) {
    extractTokens(command_line);

    resolve_command(
        target_command,
        min_param_count,
        max_param_count,
        resolution_op,
        default_help_string(target_command, min_param_count, max_param_count));
  }

  template <typename RESOLUTION_FUNCTOR>
  CommandResolutionChain& operator()(std::string const& target_command,
                                     ParamCountType min_param_count,
                                     ParamCountType max_param_count,
                                     RESOLUTION_FUNCTOR resolution_op,
                                     std::string const& custom_help) {
    if (m_resolved == true)
      return (*this);
    resolve_command(target_command,
                    min_param_count,
                    max_param_count,
                    resolution_op,
                    custom_help_string(
                        target_command, min_param_count, max_param_count, custom_help));
    return (*this);
  }

  template <typename RESOLUTION_FUNCTOR>
  CommandResolutionChain& operator()(std::string const& target_command,
                                     ParamCountType min_param_count,
                                     ParamCountType max_param_count,
                                     RESOLUTION_FUNCTOR resolution_op) {
    if (m_resolved == true)
      return (*this);
    resolve_command(
        target_command,
        min_param_count,
        max_param_count,
        resolution_op,
        default_help_string(target_command, min_param_count, max_param_count));
    return (*this);
  }

  bool is_resolved() { return m_resolved; }

 private:
  // Tag Dispatching Constructs
  struct LambdaSelector {};
  struct FunctorSelector {};
  struct RegexCommandSelector {};
  struct StandardCommandSelector {};

  void extractTokens(char const* command_line) {
    m_command_token_list = StandardTokenExtractor().extract_tokens(command_line);
    m_regex_command_token_list = RegexTokenExtractor().extract_tokens(command_line);
  }

  std::string custom_help_string(std::string const& target_command,
                                 ParamCountType min_param_count,
                                 ParamCountType max_param_count,
                                 std::string const& custom_help) {
    std::ostringstream help_stream;
    if (min_param_count == max_param_count) {
      help_stream << target_command << " incorrect number of parameters provided; need "
                  << min_param_count - 1 << " total parameter(s)\n"
                  << custom_help << std::endl;
    } else {
      help_stream << target_command
                  << " incorrect number of parameters provided; need between "
                  << min_param_count - 1 << " and " << max_param_count - 1
                  << " total parameter(s)\n"
                  << custom_help << std::endl;
    }
    return help_stream.str();
  }

  std::string default_help_string(std::string const& target_command,
                                  ParamCountType min_param_count,
                                  ParamCountType max_param_count) {
    std::ostringstream help_stream;
    if (min_param_count == max_param_count) {
      help_stream << target_command << " incorrect number of parameters provided; need "
                  << min_param_count - 1 << " total parameter(s)" << std::endl;
    } else {
      help_stream << target_command
                  << " incorrect number of parameters provided; need between "
                  << min_param_count - 1 << " and " << max_param_count - 1
                  << " total parameter(s)" << std::endl;
    }
    return help_stream.str();
  }

  template <typename RESOLUTION_MECHANISM>
  void resolve_command(std::string const& target_command,
                       ParamCountType min_param_count,
                       ParamCountType max_param_count,
                       RESOLUTION_MECHANISM resolution_op,
                       std::string const& help_info) {
    using SelectorType = typename std::conditional<
        std::is_base_of<CmdDeterminant, RESOLUTION_MECHANISM>::value,
        FunctorSelector,
        LambdaSelector>::type;

    if (m_command_token_list.empty()) {
      m_resolved = true;
    } else if (m_command_token_list[0] == target_command) {
      if (std::is_base_of<RegexCmdDeterminant, RESOLUTION_MECHANISM>::value) {
        // Regexes have an optional parameter; therefore parameter counts can vary
        // And we know the count is at least one, in this branch
        execute_functor_resolution_op(resolution_op, SelectorType());
      } else {
        if (m_command_token_list.size() < min_param_count ||
            m_command_token_list.size() > max_param_count) {
          m_output_stream << help_info << '\n';
        } else {
          execute_functor_resolution_op(resolution_op, SelectorType());
        }
      }

      m_resolved = true;
    }
  }

  template <typename RESOLUTION_LAMBDA>
  void execute_functor_resolution_op(RESOLUTION_LAMBDA resolution_op,
                                     LambdaSelector const&) {
    resolution_op(m_command_token_list);
  }

  template <typename RESOLUTION_FUNCTOR>
  void execute_functor_resolution_op(RESOLUTION_FUNCTOR resolution_op,
                                     FunctorSelector const&) {
    using CommandSelectorType = typename std::conditional<
        std::is_base_of<RegexCmdDeterminant, RESOLUTION_FUNCTOR>::value,
        RegexCommandSelector,
        StandardCommandSelector>::type;
    execute_regex_aware_resolution_op(resolution_op, CommandSelectorType());
  }

  template <typename RESOLUTION_FUNCTOR>
  void execute_regex_aware_resolution_op(RESOLUTION_FUNCTOR resolution_op,
                                         RegexCommandSelector const&) {
    resolution_op(m_regex_command_token_list, m_output_stream);
  }

  template <typename RESOLUTION_FUNCTOR>
  void execute_regex_aware_resolution_op(RESOLUTION_FUNCTOR resolution_op,
                                         StandardCommandSelector const&) {
    resolution_op(m_command_token_list, m_output_stream);
  }

  CommandTokenList m_command_token_list;
  CommandTokenList m_regex_command_token_list;
  bool m_resolved;
  std::ostream& m_output_stream;
};

#endif
