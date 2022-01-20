/*
 * Copyright (c) 2020 OmniSci, Inc.
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
   This header file provides the following C++ template functions:

     toString(const T& v) -> std::string

       Convert object v of type T to string. Pretty-printing is
       enabled for objects that types define `toString` or `to_string`
       methods.

     typeName(const T* v) -> std::string

       Return the type name of an object passed in via its pointer
       value.

    and a convenience macro `PRINT(EXPR)` that sends the string
    representation of any expression to stdout.
*/

#pragma once

#ifndef __CUDACC__

#define HAVE_TOSTRING

#ifndef _WIN32
#include <cxxabi.h>
#endif

#include <chrono>
#include <iostream>
#include <list>
#include <map>
#include <set>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "sqldefs.h"

#ifdef ENABLE_TOSTRING_RAPIDJSON
#if __has_include(<rapidjson/document.h> )
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#else
#undefine ENABLE_TOSTRING_RAPIDJSON
#endif
#endif

#ifdef ENABLE_TOSTRING_LLVM
#if __has_include(<llvm/Support/raw_os_ostream.h> )
#include <llvm/IR/Value.h>
#include <llvm/Support/raw_os_ostream.h>
#include "clang/Driver/Job.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Option/ArgList.h"
#else
#undefine ENABLE_TOSTRING_LLVM
#endif
#endif

#include <mutex>
inline static std::mutex toString_PRINT_mutex;

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define PRINT(...)                                                                      \
  {                                                                                     \
    std::lock_guard<std::mutex> print_lock(toString_PRINT_mutex);                       \
    std::cout << std::hex                                                               \
              << ((std::hash<std::thread::id>()(std::this_thread::get_id())) & 0xffff)  \
              << std::dec << " [" << __FILENAME__ << ":" << __func__ << "#" << __LINE__ \
              << "]: " #__VA_ARGS__ "=" << ::toString(std::make_tuple(__VA_ARGS__))     \
              << std::endl                                                              \
              << std::flush;                                                            \
  }

template <typename T>
std::string typeName(const T* v) {
  std::stringstream stream;
  int status;
#ifdef _WIN32
  stream << std::string(typeid(T).name());
#else
  char* demangled = abi::__cxa_demangle(typeid(T).name(), 0, 0, &status);
  stream << std::string(demangled);
  free(demangled);
#endif
  return stream.str();
}

template <typename T, typename... Args>
std::string typeName(T (*v)(Args... args)) {
  std::stringstream stream;
  int status;
#ifdef _WIN32
  stream << std::string(typeid(v).name());
#else
  char* demangled = abi::__cxa_demangle(typeid(v).name(), 0, 0, &status);
  stream << std::string(demangled);
  free(demangled);
#endif
  stream << "@0x" << std::hex << (uintptr_t)(reinterpret_cast<const void*>(v));
  return stream.str();
}

namespace {

template <typename T, typename = void>
struct has_toString : std::false_type {};
template <typename T>
struct has_toString<T, decltype(std::declval<T>().toString(), void())> : std::true_type {
};
template <class T>
inline constexpr bool has_toString_v = has_toString<T>::value;

template <typename T, typename = void>
struct get_has_toString : std::false_type {};
template <typename T>
struct get_has_toString<T, decltype(std::declval<T>().get()->toString(), void())>
    : std::true_type {};
template <class T>
inline constexpr bool get_has_toString_v = get_has_toString<T>::value;

#ifdef ENABLE_TOSTRING_to_string
template <typename T, typename = void>
struct has_to_string : std::false_type {};
template <typename T>
struct has_to_string<T, decltype(std::declval<T>().to_string(), void())>
    : std::true_type {};
template <class T>
inline constexpr bool has_to_string_v = has_to_string<T>::value;
#endif

#ifdef ENABLE_TOSTRING_str
template <typename T, typename = void>
struct has_str : std::false_type {};
template <typename T>
struct has_str<T, decltype(std::declval<T>().str(), void())> : std::true_type {};
template <class T>
inline constexpr bool has_str_v = has_str<T>::value;
#endif

template <typename T, typename = void>
struct has_printTo : std::false_type {};
template <typename T>
struct has_printTo<T,
                   decltype(std::declval<T>().printTo(std::declval<std::ostream&>()),
                            void())> : std::true_type {};
template <class T>
inline constexpr bool has_printTo_v = has_printTo<T>::value;

}  // namespace

template <typename T>
std::string toString(const T& v) {
  if constexpr (std::is_same_v<T, std::string>) {
    return "\"" + v + "\"";
  } else if constexpr (std::is_same_v<T, std::string_view>) {
    return "\"" + std::string(v) + "\"";
#ifdef ENABLE_TOSTRING_RAPIDJSON
  } else if constexpr (std::is_same_v<T, rapidjson::Value>) {
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    v.Accept(writer);
    return buffer.GetString();
#endif
#ifdef ENABLE_TOSTRING_LLVM
  } else if constexpr (std::is_same_v<T, llvm::Module>) {
    std::string type_str;
    llvm::raw_string_ostream rso(type_str);
    v.print(rso, nullptr);
    return "(" + rso.str() + ")";
  } else if constexpr (std::is_same_v<T, llvm::Function>) {
    std::string type_str;
    llvm::raw_string_ostream rso(type_str);
    v.print(rso, nullptr);
    return "(" + rso.str() + ")";
  } else if constexpr (std::is_same_v<T, llvm::Value>) {
    std::string type_str;
    llvm::raw_string_ostream rso(type_str);
    v.print(rso);
    return "(" + rso.str() + ")";
  } else if constexpr (std::is_same_v<T, llvm::Argument>) {
    std::string type_str;
    llvm::raw_string_ostream rso(type_str);
    v.print(rso);
    return "(" + rso.str() + ")";
  } else if constexpr (std::is_same_v<T, llvm::Type>) {
    std::string type_str;
    llvm::raw_string_ostream rso(type_str);
    v.print(rso);
    return "(" + rso.str() + ")";
  } else if constexpr (std::is_same_v<T, llvm::Triple>) {
    return v.str();
  } else if constexpr (std::is_same_v<T, llvm::opt::ArgStringList>) {
    std::string r;
    for (unsigned i = 0; i < v.size(); i++) {
      if (i) {
        r += ", ";
      }
      r += v[i];
    }
    return "[" + r + "]";
  } else if constexpr (std::is_same_v<T, llvm::opt::DerivedArgList>) {  // NOLINT
    std::string r;
    for (unsigned i = 0; i < v.getNumInputArgStrings(); i++) {
      if (i) {
        r += ", ";
      }
      r += v.getArgString(i);
    }
    return "[" + r + "]";
  } else if constexpr (std::is_same_v<T, clang::driver::JobList>) {  // NOLINT
    std::string type_str;
    llvm::raw_string_ostream rso(type_str);
    v.Print(rso, nullptr, true);
    return rso.str();
#endif
  } else if constexpr (std::is_same_v<T, bool>) {  // NOLINT
    return v ? "True" : "False";
  } else if constexpr (std::is_arithmetic_v<T>) {  // NOLINT
    return std::to_string(v);
#ifdef ENABLE_TOSTRING_str
  } else if constexpr (has_str_v<T>) {  // NOLINT
    return v.str();
#endif
#ifdef ENABLE_TOSTRING_to_string
  } else if constexpr (has_to_string_v<T>) {  // NOLINT
    return v.to_string();
#endif
  } else if constexpr (has_toString_v<T>) {  // NOLINT
    return v.toString();
  } else if constexpr (get_has_toString_v<T>) {
    auto ptr = v.get();
    return (ptr == NULL ? "NULL" : "&" + ptr->toString());
  } else if constexpr (std::is_same_v<T, void*>) {
    std::ostringstream ss;
    ss << std::hex << (uintptr_t)v;
    return "0x" + ss.str();
  } else if constexpr (std::is_same_v<
                           T,
                           std::chrono::time_point<std::chrono::system_clock>>) {
    std::string s(30, '\0');
    auto converted_v = (std::chrono::time_point<std::chrono::system_clock>)v;
    std::time_t ts = std::chrono::system_clock::to_time_t(v);
    std::strftime(&s[0], s.size(), "%Y-%m-%d %H:%M:%S", std::localtime(&ts));
    return s + "." +
           std::to_string((converted_v.time_since_epoch().count() / 1000) % 1000000);
  } else if constexpr (std::is_same_v<T, JoinType>) {
    switch (v) {
      case JoinType::INNER:
        return "INNER";
      case JoinType::LEFT:
        return "LEFT";
      case JoinType::SEMI:
        return "SEMI";
      case JoinType::ANTI:
        return "ANTI";
      case JoinType::INVALID:
        return "INVALID";
    }
    UNREACHABLE();
    return "";
  } else if constexpr (std::is_pointer_v<T>) {
    return (v == NULL ? "NULL" : "&" + toString(*v));
  } else if constexpr (has_printTo_v<T>) {
    std::ostringstream ss;
    v.printTo(ss);
    return ss.str();
  } else {
    return typeName(&v);
  }
}

template <typename T1, typename T2>
std::string toString(const std::pair<T1, T2>& v) {
  return "(" + toString(v.first) + ", " + toString(v.second) + ")";
}

template <typename T>
std::string toString(const std::vector<T>& v) {
  auto result = std::string("[");
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) {
      result += ", ";
    }
    result += toString(v[i]);
  }
  result += "]";
  return result;
}

template <typename T1, typename T2>
std::string toString(const std::unordered_map<T1, T2>& v) {
  auto result = std::string("{");
  size_t i = 0;
  for (const auto& p : v) {
    if (i) {
      result += ", ";
    }
    result += toString(p);
    i++;
  }
  result += "}";
  return result;
}

template <typename T1, typename T2>
std::string toString(const std::map<T1, T2>& v) {
  auto result = std::string("{");
  size_t i = 0;
  for (const auto& p : v) {
    if (i) {
      result += ", ";
    }
    result += toString(p);
    i++;
  }
  result += "}";
  return result;
}

template <typename T>
std::string toString(const std::list<T>& v) {
  auto result = std::string("[");
  size_t i = 0;
  for (const auto& p : v) {
    if (i) {
      result += ", ";
    }
    result += toString(p);
    i++;
  }
  result += "]";
  return result;
}

template <typename T>
std::string toString(const std::unordered_set<T>& v) {
  auto result = std::string("{");
  size_t i = 0;
  for (const auto& p : v) {
    if (i) {
      result += ", ";
    }
    result += toString(p);
    i++;
  }
  result += "}";
  return result;
}

template <typename T>
std::string toString(const std::set<T>& v) {
  auto result = std::string("{");
  size_t i = 0;
  for (const auto& p : v) {
    if (i) {
      result += ", ";
    }
    result += toString(p);
    i++;
  }
  result += "}";
  return result;
}

template <typename... Ts, size_t... Is>
std::string toStringImpl(const std::tuple<Ts...>& t,
                         const std::index_sequence<0, Is...>) {
  return (toString(std::get<0>(t)) + ... + (", " + toString(std::get<Is>(t))));
}

template <typename... T>
std::string toStringImpl(const std::tuple<>& t, const std::index_sequence<>) {
  return {};
}

template <typename... Ts>
std::string toString(const std::tuple<Ts...>& t) {
  return "(" + toStringImpl(t, std::index_sequence_for<Ts...>{}) + ")";
}

#endif  // ifndef __CUDACC__
