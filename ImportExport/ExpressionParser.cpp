/*
 * Copyright 2022 OmniSci, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * @file ExpressionParser.cpp
 * @author Simon Eves <simon.eves@omnisci.com>
 * @brief General Expression Parser using muparserx
 */

#include "ImportExport/ExpressionParser.h"

#include <regex>

#if defined(_MSC_VER)
#include <codecvt>
#include <locale>
#endif

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

#include <muparserx/mpParser.h>

#include "Logger/Logger.h"
#include "Shared/StringTransform.h"

namespace import_export {

namespace {

std::string ms_to_ss(const mup::string_type& s) {
#if defined(_MSC_VER)
  std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
  return converter.to_bytes(s);
#else
  return s;
#endif
}

mup::string_type ss_to_ms(const std::string& s) {
#if defined(_MSC_VER)
  std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
  return converter.from_bytes(s);
#else
  return s;
#endif
}

#define VALIDATE_ARG_TYPE(arg, t2)    \
  if (args[arg]->GetType() != t2) {   \
    mup::ErrorContext err;            \
    err.Errc = mup::ecINVALID_TYPE;   \
    err.Type1 = args[arg]->GetType(); \
    err.Type2 = t2;                   \
    err.Ident = GetIdent();           \
    throw mup::ParserError(err);      \
  }

#define THROW_INVALID_PARAMETER(arg, what)                                  \
  mup::ErrorContext err;                                                    \
  err.Errc = mup::ecINVALID_PARAMETER;                                      \
  err.Arg = arg;                                                            \
  err.Ident = GetIdent() + ss_to_ms(" (") + ss_to_ms(what) + ss_to_ms(")"); \
  throw mup::ParserError(err);

#define THROW_INVALID_PARAMETER_COUNT()           \
  mup::ErrorContext err;                          \
  err.Errc = mup::ecINVALID_NUMBER_OF_PARAMETERS; \
  err.Ident = GetIdent();                         \
  throw mup::ParserError(err);

class Function_substr : public mup::ICallback {
 public:
  Function_substr() : mup::ICallback(mup::cmFUNC, _T("substr"), -1) {}
  const mup::char_type* GetDesc() const final {
    return _T("return a substring of a string");
  };
  mup::IToken* Clone() const final { return new Function_substr(*this); };
  void Eval(mup::ptr_val_type& ret, const mup::ptr_val_type* args, int argc) final {
    if (argc < 2 || argc > 3) {
      THROW_INVALID_PARAMETER_COUNT();
    }
    VALIDATE_ARG_TYPE(0, 's');
    VALIDATE_ARG_TYPE(1, 'i');
    if (argc == 3) {
      VALIDATE_ARG_TYPE(2, 'i');
    }
    auto const text = args[0]->GetString();
    auto const start = args[1]->GetInteger();
    if (start < 1) {
      THROW_INVALID_PARAMETER(1, "bad 'start'");
    }
    if (argc == 2) {
      if (start > static_cast<int>(text.length())) {
        THROW_INVALID_PARAMETER(1, "bad 'start'");
      }
      *ret = text.substr(start - 1, std::string::npos);
    } else {
      auto const count = args[2]->GetInteger();
      if (count < 1) {
        THROW_INVALID_PARAMETER(2, "bad 'count'");
      } else if ((start - 1) + count > static_cast<int>(text.length())) {
        THROW_INVALID_PARAMETER(2, "bad 'start'/'count'");
      }
      *ret = text.substr(start - 1, count);
    }
  }
};

class Function_regex_match : public mup::ICallback {
 public:
  Function_regex_match() : mup::ICallback(mup::cmFUNC, _T("regex_match"), 2) {}
  const mup::char_type* GetDesc() const final {
    return _T("return a regex-matched section of a string");
  };
  mup::IToken* Clone() const final { return new Function_regex_match(*this); };
  void Eval(mup::ptr_val_type& ret, const mup::ptr_val_type* args, int argc) final {
    CHECK_EQ(argc, 2);
    VALIDATE_ARG_TYPE(0, 's');
    VALIDATE_ARG_TYPE(1, 's');
    auto const text = ms_to_ss(args[0]->GetString());
    auto const pattern = ms_to_ss(args[1]->GetString());
    try {
      std::regex regex(pattern, std::regex_constants::extended);
      std::smatch match;
      std::regex_match(text, match, regex);
      if (match.size() != 2u) {
        throw std::runtime_error("must have exactly one match");
      }
      *ret = ss_to_ms(match[1]);
    } catch (std::runtime_error& e) {
      THROW_INVALID_PARAMETER(2, e.what());
    }
  }
};

class Function_split_part : public mup::ICallback {
 public:
  Function_split_part() : mup::ICallback(mup::cmFUNC, _T("split_part"), 3) {}
  const mup::char_type* GetDesc() const final {
    return _T("split a string by a given separator, then return the nth token");
  };
  mup::IToken* Clone() const final { return new Function_split_part(*this); };
  void Eval(mup::ptr_val_type& ret, const mup::ptr_val_type* args, int argc) final {
    CHECK_EQ(argc, 3);
    VALIDATE_ARG_TYPE(0, 's');
    VALIDATE_ARG_TYPE(1, 's');
    VALIDATE_ARG_TYPE(2, 'i');
    auto const text = ms_to_ss(args[0]->GetString());
    auto const delimiter = ms_to_ss(args[1]->GetString());
    auto n = args[2]->GetInteger();
    try {
      std::vector<std::string> tokens;
      // split on exact delimiter (cannot use boost::split)
      size_t start{0u}, end{0u};
      while (end != std::string::npos) {
        end = text.find(delimiter, start);
        tokens.push_back(text.substr(start, end - start));
        start = end + delimiter.length();
      }
      if (tokens.size() == 0u) {
        throw std::runtime_error("failed to split");
      }
      int index{0};
      if (n < 0) {
        // reverse index (-1 = last token)
        index = static_cast<int>(tokens.size()) + n;
      } else {
        // forward index (1 = first token)
        index = n - 1;
      }
      if (index < 0 || index >= static_cast<int>(tokens.size())) {
        throw std::runtime_error("bad token index");
      }
      *ret = ss_to_ms(tokens[index]);
    } catch (std::runtime_error& e) {
      THROW_INVALID_PARAMETER(1, e.what());
    }
  }
};

class Function_int : public mup::ICallback {
 public:
  Function_int() : mup::ICallback(mup::cmFUNC, _T("int"), 1) {}
  const mup::char_type* GetDesc() const final { return _T("cast a value to an int"); };
  mup::IToken* Clone() const final { return new Function_int(*this); };
  void Eval(mup::ptr_val_type& ret, const mup::ptr_val_type* args, int argc) final {
    CHECK_EQ(argc, 1);
    switch (args[0]->GetType()) {
      case 'i':
        *ret = args[0]->GetInteger();
        break;
      case 'f':
        *ret = static_cast<mup::int_type>(args[0]->GetFloat());
        break;
      case 's':
        *ret = static_cast<mup::int_type>(std::stoll(ms_to_ss(args[0]->GetString())));
        break;
      case 'b':
        *ret = args[0]->GetBool() ? static_cast<mup::int_type>(1)
                                  : static_cast<mup::int_type>(0);
        break;
      default: {
        THROW_INVALID_PARAMETER(0, "unsupported type");
      }
    }
  }
};

class Function_float : public mup::ICallback {
 public:
  Function_float() : mup::ICallback(mup::cmFUNC, _T("float"), 1) {}
  const mup::char_type* GetDesc() const final { return _T("cast a value to a float"); };
  mup::IToken* Clone() const final { return new Function_float(*this); };
  void Eval(mup::ptr_val_type& ret, const mup::ptr_val_type* args, int argc) final {
    CHECK_EQ(argc, 1);
    switch (args[0]->GetType()) {
      case 'i':
        *ret = static_cast<mup::float_type>(args[0]->GetInteger());
        break;
      case 'f':
        *ret = args[0]->GetFloat();
        break;
      case 's':
        *ret = static_cast<mup::float_type>(std::stod(ms_to_ss(args[0]->GetString())));
        break;
      default: {
        THROW_INVALID_PARAMETER(0, "unsupported type");
      }
    }
  }
};

class Function_double : public mup::ICallback {
 public:
  Function_double() : mup::ICallback(mup::cmFUNC, _T("double"), 1) {}
  const mup::char_type* GetDesc() const final { return _T("cast a value to a double"); };
  mup::IToken* Clone() const final { return new Function_double(*this); };
  void Eval(mup::ptr_val_type& ret, const mup::ptr_val_type* args, int argc) final {
    CHECK_EQ(argc, 1);
    switch (args[0]->GetType()) {
      case 'i':
        *ret = static_cast<mup::float_type>(args[0]->GetInteger());
        break;
      case 'f':
        *ret = args[0]->GetFloat();
        break;
      case 's':
        *ret = static_cast<mup::float_type>(std::stod(ms_to_ss(args[0]->GetString())));
        break;
      default: {
        THROW_INVALID_PARAMETER(0, "unsupported type");
      }
    }
  }
};

class Function_str : public mup::ICallback {
 public:
  Function_str() : mup::ICallback(mup::cmFUNC, _T("str"), 1) {}
  const mup::char_type* GetDesc() const final { return _T("cast a value to a string"); };
  mup::IToken* Clone() const final { return new Function_str(*this); };
  void Eval(mup::ptr_val_type& ret, const mup::ptr_val_type* args, int argc) final {
    CHECK_EQ(argc, 1);
    switch (args[0]->GetType()) {
      case 'i':
      case 'f':
        *ret = args[0]->ToString();
        break;
      case 's':
        *ret = args[0]->GetString();
        break;
      case 'b':
        *ret = args[0]->GetBool() ? ss_to_ms("true") : ss_to_ms("false");
        break;
      default: {
        THROW_INVALID_PARAMETER(0, "unsupported type");
      }
    }
  }
};

class Function_bool : public mup::ICallback {
 public:
  Function_bool() : mup::ICallback(mup::cmFUNC, _T("bool"), 1) {}
  const mup::char_type* GetDesc() const final { return _T("cast a value to a boolean"); };
  mup::IToken* Clone() const final { return new Function_bool(*this); };
  void Eval(mup::ptr_val_type& ret, const mup::ptr_val_type* args, int argc) final {
    CHECK_EQ(argc, 1);
    switch (args[0]->GetType()) {
      case 'i':
        *ret = args[0]->GetInteger() != 0;
        break;
      case 's': {
        auto const s = strip(to_lower(ms_to_ss(args[0]->GetString())));
        if (s == "true" || s == "t" || s == "1") {
          *ret = true;
        } else if (s == "false" || s == "f" || s == "0") {
          *ret = false;
        } else {
          THROW_INVALID_PARAMETER(0, s.c_str());
        }
      } break;
      case 'b':
        *ret = args[0]->GetBool();
        break;
      default: {
        THROW_INVALID_PARAMETER(0, "unsupported type");
      }
    }
  }
};

class Operator_not : public mup::IOprtInfix {
 public:
  Operator_not() : mup::IOprtInfix(_T("not"), mup::prINFIX) {}
  const mup::char_type* GetDesc() const final { return _T("bool inversion operator"); }
  mup::IToken* Clone() const final { return new Operator_not(*this); }
  void Eval(mup::ptr_val_type& ret, const mup::ptr_val_type* args, int argc) final {
    CHECK_EQ(argc, 1);
    VALIDATE_ARG_TYPE(0, 'b');
    *ret = !(args[0]->GetBool());
  }
};

mup::Value evaluate(mup::ParserX* parser) {
  mup::Value result;
  try {
    result = parser->Eval();
  } catch (mup::ParserError& err) {
    throw std::runtime_error("Parser Error: " + ms_to_ss(err.GetMsg()));
  } catch (std::exception& err) {
    throw std::runtime_error("Unexpected muparserx Error: " + std::string(err.what()));
  }
  return result;
}

}  // namespace

void ExpressionParser::ParserDeleter::operator()(mup::ParserX* parser) {
  delete parser;
}

ExpressionParser::ExpressionParser()
    : parser_{new mup::ParserX(mup::pckCOMMON | mup::pckUNIT | mup::pckNON_COMPLEX |
                               mup::pckSTRING)} {
  // custom operators and functions
  parser_->DefineFun(new Function_substr());
  parser_->DefineFun(new Function_regex_match());
  parser_->DefineFun(new Function_split_part());
  parser_->DefineFun(new Function_int());
  parser_->DefineFun(new Function_float());
  parser_->DefineFun(new Function_double());
  parser_->DefineFun(new Function_str());
  parser_->DefineFun(new Function_bool());
  parser_->DefineInfixOprt(new Operator_not());
}

void ExpressionParser::setStringConstant(const std::string& name,
                                         const std::string& value) {
  parser_->DefineConst(ss_to_ms(name), ss_to_ms(value));
}

void ExpressionParser::setIntConstant(const std::string& name, const int value) {
  parser_->DefineConst(ss_to_ms(name), static_cast<mup::int_type>(value));
}

void ExpressionParser::setExpression(const std::string& expression) {
  parser_->SetExpr(ss_to_ms(expression));
}

std::string ExpressionParser::evalAsString() {
  auto result = evaluate(parser_.get());
  if (result.GetType() != 's') {
    throw std::runtime_error("Expression is not a string");
  }
  return ms_to_ss(result.GetString());
}

int ExpressionParser::evalAsInt() {
  auto result = evaluate(parser_.get());
  if (result.GetType() != 'i') {
    throw std::runtime_error("Expression is not an int");
  }
  return static_cast<int>(result.GetInteger());
}

double ExpressionParser::evalAsDouble() {
  auto result = evaluate(parser_.get());
  if (result.GetType() != 'f') {
    throw std::runtime_error("Expression is not a float/double");
  }
  return static_cast<double>(result.GetFloat());
}

bool ExpressionParser::evalAsBool() {
  auto result = evaluate(parser_.get());
  if (result.GetType() != 'b') {
    throw std::runtime_error("Expression is not a boolean");
  }
  return result.GetBool();
}

}  // namespace import_export
