#ifndef STRING_TO_DATUM_H
#define STRING_TO_DATUM_H

#include <Utils/StringConversions.h>
#include <boost/utility/string_view.hpp>
#include <cinttypes>
#include "../Shared/StringToDatum.h"
#include "DateConverters.h"
#include "TimeGM.h"
#include "sqltypes.h"

template <typename String>
Datum StringToDatum(const String& s, SQLTypeInfo& ti) {
  Datum d;
  switch (ti.get_type()) {
    case kARRAY:
      break;
    case kBOOLEAN:
      if (s == "t" || s == "T" || s == "1" || boost::iequals(s, "TRUE")) {
        d.boolval = true;
      } else if (s == "f" || s == "F" || s == "0" || boost::iequals(s, "FALSE")) {
        d.boolval = false;
      } else {
        throw std::runtime_error("Invalid string for boolean " +
                                 StringConversions::to_string(s));
      }
      break;
    case kNUMERIC:
    case kDECIMAL:
      d.bigintval = parse_numeric(s, ti);
      break;
    case kBIGINT:
      d.bigintval = StringConversions::strtol(s);
      break;
    case kINT:
      d.intval = StringConversions::strtol(s);
      break;
    case kSMALLINT:
      d.smallintval = StringConversions::strtol(s);
      break;
    case kTINYINT:
      d.tinyintval = StringConversions::strtol(s);
      break;
    case kFLOAT:
      d.floatval = StringConversions::strtof(s);
      break;
    case kDOUBLE:
      d.doubleval = StringConversions::strtod(s);
      break;
    case kTIME: {
      // @TODO handle fractional seconds
      std::tm tm_struct = {0};
      auto st = StringConversions::to_string(s);
      if (!strptime(st.c_str(), "%T %z", &tm_struct) &&
          !strptime(st.c_str(), "%T", &tm_struct) &&
          !strptime(st.c_str(), "%H%M%S", &tm_struct) &&
          !strptime(st.c_str(), "%R", &tm_struct)) {
        throw std::runtime_error("Invalid time string " + st);
      }
      tm_struct.tm_mday = 1;
      tm_struct.tm_mon = 0;
      tm_struct.tm_year = 70;
      tm_struct.tm_wday = tm_struct.tm_yday = tm_struct.tm_isdst = tm_struct.tm_gmtoff =
          0;
      d.bigintval = static_cast<int64_t>(TimeGM::instance().my_timegm(&tm_struct));
      break;
    }
    case kTIMESTAMP: {
      auto st = StringConversions::to_string(s);
      std::tm tm_struct = {0};
      // not sure in advance if it is used so need to zero before processing
      tm_struct.tm_gmtoff = 0;
      char* tp;
      // try ISO8601 date first
      tp = strptime(st.c_str(), "%Y-%m-%d", &tm_struct);
      if (!tp) {
        tp = strptime(st.c_str(), "%m/%d/%Y", &tm_struct);  // accept American date
      }
      if (!tp) {
        tp = strptime(st.c_str(), "%d-%b-%y", &tm_struct);  // accept 03-Sep-15
      }
      if (!tp) {
        tp = strptime(st.c_str(), "%d/%b/%Y", &tm_struct);  // accept 03/Sep/2015
      }
      if (!tp) {
        try {
          d.bigintval = static_cast<int64_t>(std::stoll(st));
          break;
        } catch (const std::invalid_argument& ia) {
          throw std::runtime_error("Invalid timestamp string " + st);
        }
      }
      if (*tp == 'T' || *tp == ' ' || *tp == ':') {
        tp++;
      } else {
        throw std::runtime_error("Invalid timestamp break string " + st);
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
        throw std::runtime_error("Invalid timestamp time string " +
                                 StringConversions::to_string(s));
      }
      tm_struct.tm_wday = tm_struct.tm_yday = tm_struct.tm_isdst = 0;
      // handle fractional seconds
      if (ti.get_dimension() > 0) {  // check for precision
        time_t fsc;
        if (*p == '.') {
          p++;
          uint64_t frac_num = 0;
          int ntotal = 0;
          sscanf(p, "%" SCNu64 "%n", &frac_num, &ntotal);
          fsc = TimeGM::instance().parse_fractional_seconds(frac_num, ntotal, ti);
        } else if (*p == '\0') {
          fsc = 0;
        } else {  // check for misleading/unclear syntax
          throw std::runtime_error("Unclear syntax for leading fractional seconds: " +
                                   std::string(p));
        }
        d.bigintval =
            static_cast<int64_t>(TimeGM::instance().my_timegm(&tm_struct, fsc, ti));
      } else {  // default timestamp(0) precision
        d.bigintval = static_cast<int64_t>(TimeGM::instance().my_timegm(&tm_struct));
        if (*p == '.') {
          p++;
        }
      }
      if (*p != '\0') {
        uint32_t hour = 0;
        sscanf(tp, "%u", &hour);
        d.bigintval = static_cast<int64_t>(TimeGM::instance().parse_meridians(
            static_cast<time_t>(d.bigintval), p, hour, ti));
        break;
      }
      break;
    }
    case kDATE: {
      auto st = StringConversions::to_string(s);
      std::tm tm_struct = {0};
      // not sure in advance if it is used so need to zero before processing
      tm_struct.tm_gmtoff = 0;
      char* tp;
      // try ISO8601 date first
      tp = strptime(st.c_str(), "%Y-%m-%d", &tm_struct);
      if (!tp) {
        tp = strptime(st.c_str(), "%m/%d/%Y", &tm_struct);  // accept American date
      }
      if (!tp) {
        tp = strptime(st.c_str(), "%d-%b-%y", &tm_struct);  // accept 03-Sep-15
      }
      if (!tp) {
        tp = strptime(st.c_str(), "%d/%b/%Y", &tm_struct);  // accept 03/Sep/2015
      }
      if (!tp) {
        try {
          d.bigintval = static_cast<int64_t>(std::stoll(st));
          break;
        } catch (const std::invalid_argument& ia) {
          throw std::runtime_error("Invalid date string " + st);
        }
      }
      tm_struct.tm_sec = tm_struct.tm_min = tm_struct.tm_hour = 0;
      tm_struct.tm_wday = tm_struct.tm_yday = tm_struct.tm_isdst = tm_struct.tm_gmtoff =
          0;
      d.bigintval = static_cast<int64_t>(TimeGM::instance().my_timegm(&tm_struct));
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

template <typename String>
int64_t parse_numeric(const String& s, SQLTypeInfo& ti) {
  const static char* zero = "0";
  assert(s.length() <= 30);
  size_t dot = s.find_first_of('.', 0);
  String before_dot;
  String after_dot;
  if (dot != std::string::npos) {
    // make .99 as 0.99, or std::stoll below throws exception 'std::invalid_argument'
    before_dot = (0 == dot) ? zero : s.substr(0, dot);
    after_dot = s.substr(dot + 1);
  } else {
    before_dot = s;
    after_dot = zero;
  }
  const bool is_negative = before_dot.find_first_of('-', 0) != std::string::npos;
  const int64_t sign = is_negative ? -1 : 1;
  int64_t result;
  result = std::abs(StringConversions::strtol(before_dot));
  int64_t fraction = 0;
  const size_t before_dot_digits = before_dot.length() - (is_negative ? 1 : 0);
  if (!after_dot.empty()) {
    fraction = StringConversions::strtol(after_dot);
  }
  if (ti.get_dimension() == 0) {
    // set the type info based on the literal string
    ti.set_scale(after_dot.length());
    ti.set_dimension(before_dot_digits + ti.get_scale());
    ti.set_notnull(false);
  } else {
    if (before_dot_digits + ti.get_scale() > static_cast<size_t>(ti.get_dimension())) {
      throw std::runtime_error("numeric value " + StringConversions::to_string(s) +
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

#endif