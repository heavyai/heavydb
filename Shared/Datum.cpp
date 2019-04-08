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

#include <glog/logging.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include "StringTransform.h"

#include "DateConversions.h"
#include "TimeGM.h"
#include "sqltypes.h"

int64_t parse_numeric(const std::string& s, SQLTypeInfo& ti) {
  assert(s.length() <= 20);
  size_t dot = s.find_first_of('.', 0);
  std::string before_dot;
  std::string after_dot;
  if (dot != std::string::npos) {
    // make .99 as 0.99, or std::stoll below throws exception 'std::invalid_argument'
    before_dot = (0 == dot) ? "0" : s.substr(0, dot);
    after_dot = s.substr(dot + 1);
  } else {
    before_dot = s;
    after_dot = "0";
  }
  const bool is_negative = before_dot.find_first_of('-', 0) != std::string::npos;
  const int64_t sign = is_negative ? -1 : 1;
  int64_t result;
  result = std::abs(std::stoll(before_dot));
  int64_t fraction = 0;
  const size_t before_dot_digits = before_dot.length() - (is_negative ? 1 : 0);
  if (!after_dot.empty()) {
    fraction = std::stoll(after_dot);
  }
  if (ti.get_dimension() == 0) {
    // set the type info based on the literal string
    ti.set_scale(after_dot.length());
    ti.set_dimension(before_dot_digits + ti.get_scale());
    ti.set_notnull(false);
  } else {
    if (before_dot_digits + ti.get_scale() > static_cast<size_t>(ti.get_dimension())) {
      throw std::runtime_error("numeric value " + s +
                               " exceeds the maximum precision of " +
                               std::to_string(ti.get_dimension()));
    }
    for (ssize_t i = 0; i < static_cast<ssize_t>(after_dot.length()) - ti.get_scale();
         i++) {
      fraction /= 10;  // truncate the digits after decimal point.
    }
  }
  // the following loop can be made more efficient if needed
  for (int i = 0; i < ti.get_scale(); i++) {
    result *= 10;
  }
  if (result < 0) {
    result -= fraction;
  } else {
    result += fraction;
  }
  return result * sign;
}

/*
 * @brief convert string to a datum
 */
Datum StringToDatum(const std::string& s, SQLTypeInfo& ti) {
  Datum d;
  switch (ti.get_type()) {
    case kARRAY:
      break;
    case kBOOLEAN:
      if (s == "t" || s == "T" || s == "1" || to_upper(s) == "TRUE") {
        d.boolval = true;
      } else if (s == "f" || s == "F" || s == "0" || to_upper(s) == "FALSE") {
        d.boolval = false;
      } else {
        throw std::runtime_error("Invalid string for boolean " + s);
      }
      break;
    case kNUMERIC:
    case kDECIMAL:
      d.bigintval = parse_numeric(s, ti);
      break;
    case kBIGINT:
      d.bigintval = std::stoll(s);
      break;
    case kINT:
      d.intval = std::stoi(s);
      break;
    case kSMALLINT:
      d.smallintval = std::stoi(s);
      break;
    case kTINYINT:
      d.tinyintval = std::stoi(s);
      break;
    case kFLOAT:
      d.floatval = std::stof(s);
      break;
    case kDOUBLE:
      d.doubleval = std::stod(s);
      break;
    case kTIME: {
      // @TODO handle fractional seconds
      std::tm tm_struct = {0};
      if (!strptime(s.c_str(), "%T %z", &tm_struct) &&
          !strptime(s.c_str(), "%T", &tm_struct) &&
          !strptime(s.c_str(), "%H%M%S", &tm_struct) &&
          !strptime(s.c_str(), "%R", &tm_struct)) {
        throw std::runtime_error("Invalid time string " + s);
      }
      tm_struct.tm_mday = 1;
      tm_struct.tm_mon = 0;
      tm_struct.tm_year = 70;
      tm_struct.tm_wday = tm_struct.tm_yday = tm_struct.tm_isdst = tm_struct.tm_gmtoff =
          0;
      d.timeval = TimeGM::instance().my_timegm(&tm_struct);
      break;
    }
    case kTIMESTAMP: {
      std::tm tm_struct = {0};
      // not sure in advance if it is used so need to zero before processing
      tm_struct.tm_gmtoff = 0;
      char* tp;
      // try ISO8601 date first
      tp = strptime(s.c_str(), "%Y-%m-%d", &tm_struct);
      if (!tp) {
        tp = strptime(s.c_str(), "%m/%d/%Y", &tm_struct);  // accept American date
      }
      if (!tp) {
        tp = strptime(s.c_str(), "%d-%b-%y", &tm_struct);  // accept 03-Sep-15
      }
      if (!tp) {
        tp = strptime(s.c_str(), "%d/%b/%Y", &tm_struct);  // accept 03/Sep/2015
      }
      if (!tp) {
        try {
          d.timeval = std::stoll(s);
          break;
        } catch (const std::invalid_argument& ia) {
          throw std::runtime_error("Invalid timestamp string " + s);
        }
      }
      if (*tp == 'T' || *tp == ' ' || *tp == ':') {
        tp++;
      } else {
        throw std::runtime_error("Invalid timestamp break string " + s);
      }
      // now parse the time
      char* p = strptime(tp, "%T %z", &tm_struct);
      if (!p) {
        p = strptime(tp, "%T", &tm_struct);
      }
      if (!p) {
        p = strptime(tp, "%H%M%S", &tm_struct);
      }
      if (!p) {
        p = strptime(tp, "%R", &tm_struct);
      }
      if (!p) {
        // check for weird customer format
        // remove decimal seconds from string if there is a period followed by a number
        char* startptr = nullptr;
        char* endptr;
        // find last decimal in string
        int loop = strlen(tp);
        while (loop > 0) {
          if (tp[loop] == '.') {
            // found last period
            startptr = &tp[loop];
            break;
          }
          loop--;
        }
        if (startptr) {
          // look for space
          endptr = strchr(startptr, ' ');
          if (endptr) {
            // ok we found a start and and end
            // remove the decimal portion
            // will need to capture this for later
            memmove(startptr, endptr, strlen(endptr) + 1);
          }
        }
        p = strptime(
            tp, "%I . %M . %S %p", &tm_struct);  // customers weird '.' separated date
      }
      if (!p) {
        throw std::runtime_error("Invalid timestamp time string " + s);
      }
      tm_struct.tm_wday = tm_struct.tm_yday = tm_struct.tm_isdst = 0;
      // handle fractional seconds
      if (ti.get_dimension() > 0) {  // check for precision
        time_t fsc;
        if (*p == '.') {
          p++;
          uint64_t frac_num = 0;
          int ntotal = 0;
          sscanf(p, "%lu%n", &frac_num, &ntotal);
          fsc = TimeGM::instance().parse_fractional_seconds(frac_num, ntotal, ti);
        } else if (*p == '\0') {
          fsc = 0;
        } else {  // check for misleading/unclear syntax
          throw std::runtime_error("Unclear syntax for leading fractional seconds: " +
                                   std::string(p));
        }
        d.timeval = TimeGM::instance().my_timegm(&tm_struct, fsc, ti);
      } else {  // default timestamp(0) precision
        d.timeval = TimeGM::instance().my_timegm(&tm_struct);
        if (*p == '.') {
          p++;
        }
      }
      if (*p != '\0') {
        uint32_t hour = 0;
        sscanf(tp, "%u", &hour);
        d.timeval = TimeGM::instance().parse_meridians(d.timeval, p, hour, ti);
        break;
      }
      break;
    }
    case kDATE: {
      std::tm tm_struct = {0};
      // not sure in advance if it is used so need to zero before processing
      tm_struct.tm_gmtoff = 0;
      char* tp;
      // try ISO8601 date first
      tp = strptime(s.c_str(), "%Y-%m-%d", &tm_struct);
      if (!tp) {
        tp = strptime(s.c_str(), "%m/%d/%Y", &tm_struct);  // accept American date
      }
      if (!tp) {
        tp = strptime(s.c_str(), "%d-%b-%y", &tm_struct);  // accept 03-Sep-15
      }
      if (!tp) {
        tp = strptime(s.c_str(), "%d/%b/%Y", &tm_struct);  // accept 03/Sep/2015
      }
      if (!tp) {
        try {
          d.timeval = ti.is_date_in_days()
                          ? DateConverters::get_epoch_days_from_seconds(std::stoll(s))
                          : std::stoll(s);
          break;
        } catch (const std::invalid_argument& ia) {
          throw std::runtime_error("Invalid date string " + s);
        }
      }
      tm_struct.tm_sec = tm_struct.tm_min = tm_struct.tm_hour = 0;
      tm_struct.tm_wday = tm_struct.tm_yday = tm_struct.tm_isdst = tm_struct.tm_gmtoff =
          0;
      d.timeval = ti.is_date_in_days() ? TimeGM::instance().my_timegm_days(&tm_struct)
                                       : TimeGM::instance().my_timegm(&tm_struct);
      break;
    }
    case kPOINT:
    case kLINESTRING:
    case kPOLYGON:
    case kMULTIPOLYGON:
      throw std::runtime_error("Internal error: geometry type in StringToDatum.");
    default:
      throw std::runtime_error("Internal error: invalid type in StringToDatum.");
  }
  return d;
}

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
      return a.timeval == b.timeval;
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
      gmtime_r(&d.timeval, &tm_struct);
      char buf[9];
      strftime(buf, 9, "%T", &tm_struct);
      return std::string(buf);
    }
    case kTIMESTAMP: {
      std::tm tm_struct{0};
      if (ti.get_dimension() > 0) {
        std::string t = std::to_string(d.timeval);
        int cp = t.length() - ti.get_dimension();
        time_t sec = std::stoll(t.substr(0, cp));
        t = t.substr(cp);
        gmtime_r(&sec, &tm_struct);
        char buf[21];
        strftime(buf, 21, "%F %T.", &tm_struct);
        return std::string(buf) += t;
      } else {
        time_t sec = d.timeval;
        gmtime_r(&sec, &tm_struct);
        char buf[20];
        strftime(buf, 20, "%F %T", &tm_struct);
        return std::string(buf);
      }
    }
    case kDATE: {
      std::tm tm_struct;
      time_t ntimeval = ti.is_date_in_days() ? d.timeval * 86400 : d.timeval;
      gmtime_r(&ntimeval, &tm_struct);
      char buf[11];
      strftime(buf, 11, "%F", &tm_struct);
      return std::string(buf);
    }
    case kINTERVAL_DAY_TIME:
      return std::to_string(d.timeval) + " ms (day-time interval)";
    case kINTERVAL_YEAR_MONTH:
      return std::to_string(d.timeval) + " month(s) (year-month interval)";
    case kTEXT:
    case kVARCHAR:
    case kCHAR:
      return *d.stringval;
    default:
      throw std::runtime_error("Internal error: invalid type in DatumToString.");
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
