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
#if __cplusplus >= 201703L

#define HAVE_TOSTRING

#include <cxxabi.h>
#include <chrono>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#else
#undefine ENABLE_TOSTRING_LLVM
#endif
#endif

#define PRINT(EXPR)                                                              \
  std::cout << __func__ << "#" << __LINE__ << ": " #EXPR "=" << ::toString(EXPR) \
            << std::endl;

template <typename T>
std::string typeName(const T* v) {
  std::stringstream stream;
  int status;
  char* demangled = abi::__cxa_demangle(typeid(T).name(), 0, 0, &status);
  stream << std::string(demangled);
  free(demangled);
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

}  // namespace

template <typename T>
std::string toString(const T& v) {
  if constexpr (std::is_same_v<T, std::string>) {
    return "\"" + v + "\"";
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
  } else if constexpr (std::is_same_v<T, llvm::Type>) {
    std::string type_str;
    llvm::raw_string_ostream rso(type_str);
    v.print(rso);
    return "(" + rso.str() + ")";
#endif
  } else if constexpr (std::is_same_v<T, bool>) {
    return v ? "True" : "False";
  } else if constexpr (std::is_arithmetic_v<T>) {
    return std::to_string(v);
#ifdef ENABLE_TOSTRING_str
  } else if constexpr (has_str_v<T>) {
    return v.str();
#endif
#ifdef ENABLE_TOSTRING_to_string
  } else if constexpr (has_to_string_v<T>) {
    return v.to_string();
#endif
  } else if constexpr (has_toString_v<T>) {
    return v.toString();
  } else if constexpr (get_has_toString_v<T>) {
    return v.get()->toString();
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
  } else if constexpr (std::is_pointer_v<T>) {
    return (v == NULL ? "NULL" : "&" + toString(*v));
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

#endif  //  __cplusplus >= 201703L
#endif  // __CUDACC__
