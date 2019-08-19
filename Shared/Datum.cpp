/*
 * Copyright 2017 MapD Technologies, Inc.
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

/**
 * @file		DatumString.cpp
 * @author	Wei Hong <wei@map-d.com>
 * @brief		Functions to convert between strings and Datum
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/

#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif
#include <cinttypes>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include "Logger.h"
#include "StringTransform.h"

#include "sqltypes.h"

bool DatumEqual(const Datum a, const Datum b, const SQLTypeInfo& ti) {
  switch (ti.get_type()) {
    case kBOOLEAN:
      return a.boolval == b.boolval;
    case kBIGINT:
    case kNUMERIC:
    case kDECIMAL:
      return a.bigintval == b.bigintval;
    case kINT:
      return a.intval == b.intval;
    case kSMALLINT:
      return a.smallintval == b.smallintval;
    case kTINYINT:
      return a.tinyintval == b.tinyintval;
    case kFLOAT:
      return a.floatval == b.floatval;
    case kDOUBLE:
      return a.doubleval == b.doubleval;
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
    case kINTERVAL_DAY_TIME:
    case kINTERVAL_YEAR_MONTH:
      return a.bigintval == b.bigintval;
    case kTEXT:
    case kVARCHAR:
    case kCHAR:
      if (ti.get_compression() == kENCODING_DICT) {
        return a.intval == b.intval;
      }
      return *a.stringval == *b.stringval;
    default:
      return false;
  }
  return false;
}

/*
 * @brief convert datum to string
 */
std::string DatumToString(Datum d, const SQLTypeInfo& ti) {
  switch (ti.get_type()) {
    case kBOOLEAN:
      if (d.boolval) {
        return "t";
      }
      return "f";
    case kNUMERIC:
    case kDECIMAL: {
      char str[ti.get_dimension() + 1];
      double v = (double)d.bigintval / pow(10, ti.get_scale());
      sprintf(str, "%*.*f", ti.get_dimension(), ti.get_scale(), v);
      return std::string(str);
    }
    case kINT:
      return std::to_string(d.intval);
    case kSMALLINT:
      return std::to_string(d.smallintval);
    case kTINYINT:
      return std::to_string(d.tinyintval);
    case kBIGINT:
      return std::to_string(d.bigintval);
    case kFLOAT:
      return std::to_string(d.floatval);
    case kDOUBLE:
      return std::to_string(d.doubleval);
    case kTIME: {
      std::tm tm_struct;
      gmtime_r(reinterpret_cast<time_t*>(&d.bigintval), &tm_struct);
      char buf[9];
      strftime(buf, 9, "%T", &tm_struct);
      return std::string(buf);
    }
    case kTIMESTAMP: {
      std::tm tm_struct{0};
      if (ti.get_dimension() > 0) {
        std::string t = std::to_string(d.bigintval);
        int cp = t.length() - ti.get_dimension();
        time_t sec = std::stoll(t.substr(0, cp));
        t = t.substr(cp);
        gmtime_r(&sec, &tm_struct);
        char buf[21];
        strftime(buf, 21, "%F %T.", &tm_struct);
        return std::string(buf) += t;
      } else {
        time_t sec = static_cast<time_t>(d.bigintval);
        gmtime_r(&sec, &tm_struct);
        char buf[20];
        strftime(buf, 20, "%F %T", &tm_struct);
        return std::string(buf);
      }
    }
    case kDATE: {
      std::tm tm_struct;
      time_t ntimeval = static_cast<time_t>(d.bigintval);
      gmtime_r(&ntimeval, &tm_struct);
      char buf[11];
      strftime(buf, 11, "%F", &tm_struct);
      return std::string(buf);
    }
    case kINTERVAL_DAY_TIME:
      return std::to_string(d.bigintval) + " ms (day-time interval)";
    case kINTERVAL_YEAR_MONTH:
      return std::to_string(d.bigintval) + " month(s) (year-month interval)";
    case kTEXT:
    case kVARCHAR:
    case kCHAR:
      return *d.stringval;
    default:
      throw std::runtime_error("Internal error: invalid type " + ti.get_type_name() +
                               " in DatumToString.");
  }
  return "";
}

SQLTypes decimal_to_int_type(const SQLTypeInfo& ti) {
  switch (ti.get_size()) {
    case 1:
      return kTINYINT;
    case 2:
      return kSMALLINT;
    case 4:
      return kINT;
    case 8:
      return kBIGINT;
    default:
      CHECK(false);
  }
  return kNULLT;
}

int64_t convert_decimal_value_to_scale(const int64_t decimal_value,
                                       const SQLTypeInfo& type_info,
                                       const SQLTypeInfo& new_type_info) {
  auto converted_decimal_value = decimal_value;
  if (new_type_info.get_scale() > type_info.get_scale()) {
    for (int i = 0; i < new_type_info.get_scale() - type_info.get_scale(); i++) {
      converted_decimal_value *= 10;
    }
  } else if (new_type_info.get_scale() < type_info.get_scale()) {
    for (int i = 0; i < type_info.get_scale() - new_type_info.get_scale(); i++) {
      if (converted_decimal_value > 0) {
        converted_decimal_value = (converted_decimal_value + 5) / 10;
      } else {
        converted_decimal_value = (converted_decimal_value - 5) / 10;
      }
    }
  }
  return converted_decimal_value;
}
