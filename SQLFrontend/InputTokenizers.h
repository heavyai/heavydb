#ifndef INPUTTOKENIZERS_H
#define INPUTTOKENIZERS_H

#include <boost/tokenizer.hpp>
#include <iostream>
#include <sstream>

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

template <typename TOKEN_LIST_TYPE>
class QuotedInputSupportParser {
 public:
  template <typename INPUT_TYPE>
  TOKEN_LIST_TYPE extract_tokens(INPUT_TYPE& input_stream, std::ostream& error_stream = std::cout) {
    using Tokenizer = boost::tokenizer<boost::escaped_list_separator<char>>;
    using TokenIterator = Tokenizer::iterator;
    TOKEN_LIST_TYPE token_list;

    std::istringstream input_resolver(input_stream);
    std::string input_token;
    if (input_resolver >> input_token) {
      token_list.push_back(input_token);  // First token
    } else {
      return token_list;  // No token, no more business, return nothing.
    }

    // The reason the tokenizing is broken into two portions is because
    // boost tokenizer thinks the backslash commands are escape
    // sequences.  They aren't.  We extract the command first, then
    // extract the portion sent to the tokenizer
    std::string second_half;
    std::getline(input_resolver, second_half);
    if (!input_resolver)
      return token_list;

    try {
      boost::escaped_list_separator<char> els("\\", " ", "\"");
      Tokenizer splitter(second_half, els);

      for (TokenIterator i = splitter.begin(); i != splitter.end(); i++) {
        if (!i->empty()) {
          token_list.push_back(*i);
        }
      }
    } catch (boost::exception& e) {
      // Silently fail; odds are this input stream will be processed by a parallel input tokenizer (like for Regex
      // commands)
    }  // Best efforts

    return token_list;
  }
};

template <typename TOKEN_LIST_TYPE>
using UnitTestOutputTokenizer = DefaultInputParser<TOKEN_LIST_TYPE>;

#endif
