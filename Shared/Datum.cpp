/**
 * @file		DatumString.cpp
 * @author	Wei Hong <wei@map-d.com>
 * @brief		Functions to convert between strings and Datum
 * 
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/

#include <glog/logging.h>
#include <string>
#include <stdexcept>
#include <cassert>
#include <cstdio>
#include <cmath>
#include "sqltypes.h"

std::string SQLTypeInfo::type_name[kSQLTYPE_LAST] = { "NULL", "BOOLEAN", "CHAR", "VARCHAR", "NUMERIC", "DECIMAL", "INTEGER", "SMALLINT", "FLOAT", "DOUBLE", "TIME", "TIMESTAMP", "BIGINT", "TEXT", "DATE", "ARRAY" };
std::string SQLTypeInfo::comp_name[kENCODING_LAST] = { "NONE", "FIXED", "RL", "DIFF", "DICT", "SPARSE" };

int64_t
parse_numeric(const std::string &s, SQLTypeInfo &ti)
{
	assert(s.length() <= 20);
	size_t dot = s.find_first_of('.', 0);
  std::string before_dot;
  std::string after_dot;
  if (dot != std::string::npos) {
    before_dot = s.substr(0, dot);
    after_dot = s.substr(dot+1);
  } else {
    before_dot = s;
    after_dot = "0";
  }
	int64_t result;
	result = std::stoll(before_dot);
	int64_t fraction = 0;
  if (!after_dot.empty())
    fraction = std::stoll(after_dot);
	if (ti.get_dimension() == 0) {
		// set the type info based on the literal string
		ti.set_scale(after_dot.length());
		ti.set_dimension(before_dot.length() + ti.get_scale());
		ti.set_notnull(false);
	} else {
		if (before_dot.length() + ti.get_scale() > static_cast<size_t>(ti.get_dimension()))
			throw std::runtime_error("numeric value " + s + " exceeds the maximum precision of " + std::to_string(ti.get_dimension()));
		for (size_t i = 0; i < after_dot.length() - static_cast<size_t>(ti.get_scale()); i++)
			fraction /= 10; // truncate the digits after decimal point.
	}
	// the following loop can be made more efficient if needed
	for (int i = 0; i < ti.get_scale(); i++)
		result *= 10;
	if (result < 0)
		result -= fraction;
	else
		result += fraction;
	return result;
}

// had to port timegm because the one on MacOS is horrendously slow.
extern time_t my_timegm(const struct tm *tm);

/*
 * @brief convert string to a datum
 */
Datum
StringToDatum(const std::string &s, SQLTypeInfo &ti)
{
	Datum d;
	switch (ti.get_type()) {
		case kBOOLEAN:
			if (s == "t" || s == "true" || s == "T" || s == "True")
				d.boolval = true;
			else if (s == "f" || s == "false" || s == "F" || s == "False")
				d.boolval = false;
			else
				throw std::runtime_error("Invalid string for boolean " + s);
			break;
		case kNUMERIC:
		case kDECIMAL:
			d.bigintval = parse_numeric(s, ti);
			break;
		case kBIGINT:
			d.bigintval = std::stoi(s);
			break;
		case kINT:
			d.intval = std::stoi(s);
			break;
		case kSMALLINT:
			d.smallintval = std::stoi(s);
			break;
		case kFLOAT:
			d.floatval = std::stof(s);
			break;
		case kDOUBLE:
			d.doubleval = std::stod(s);
			break;
		case kTIME:
			{
				// @TODO handle fractional seconds
				std::tm tm_struct;
				if (strptime(s.c_str(), "%T", &tm_struct) == nullptr &&
					  strptime(s.c_str(), "%H%M%S", &tm_struct) == nullptr)
					throw std::runtime_error("Invalid time string " + s);
			  tm_struct.tm_mday = 1;
				tm_struct.tm_mon = 0;
				tm_struct.tm_year = 70;
				tm_struct.tm_wday = tm_struct.tm_yday = tm_struct.tm_isdst = 0;
				d.timeval = my_timegm(&tm_struct);
				break;
			}
		case kTIMESTAMP:
			{
				std::tm tm_struct;
				char *tp;
				// try ISO8601 date first
				tp = strptime(s.c_str(), "%Y-%m-%d", &tm_struct);
				if (tp == nullptr)
					tp = strptime(s.c_str(), "%m/%d/%Y", &tm_struct); // accept American date
				if (tp == nullptr) {
          try {
            d.timeval = std::stoll(s);
            break;
          }
          catch (const std::invalid_argument& ia) {
            throw std::runtime_error("Invalid timestamp string " + s);
          }
        }
				if (*tp == 'T' || *tp == ' ')
					tp++;
				else
					throw std::runtime_error("Invalid timestamp string " + s);
				// now parse the time
				// @TODO handle fractional seconds
				char *p = strptime(tp, "%T", &tm_struct);
				if (p == nullptr)
					p = strptime(tp, "%H%M%S", &tm_struct);
				if (p == nullptr)
					throw std::runtime_error("Invalid timestamp string " + s);
				tm_struct.tm_wday = tm_struct.tm_yday = tm_struct.tm_isdst = 0;
				d.timeval = my_timegm(&tm_struct);
				break;
			}
		case kDATE:
			{
				std::tm tm_struct;
				if (strptime(s.c_str(), "%Y-%m-%d", &tm_struct) == nullptr &&
						strptime(s.c_str(), "%m/%d/%Y", &tm_struct) == nullptr)
					throw std::runtime_error("Invalid timestamp string " + s);
				tm_struct.tm_sec = tm_struct.tm_min = tm_struct.tm_hour = 0;
				tm_struct.tm_wday = tm_struct.tm_yday = tm_struct.tm_isdst = 0;
				d.timeval = my_timegm(&tm_struct);
				break;
			}
		default:
			throw std::runtime_error("Internal error: invalid type in StringToDatum.");
	}
	return d;
}

/*
 * @brief convert datum to string
 */
std::string
DatumToString(Datum d, const SQLTypeInfo &ti)
{
	switch (ti.get_type()) {
		case kBOOLEAN:
			if (d.boolval)
				return "t";
			return "f";
		case kNUMERIC:
		case kDECIMAL:
			{
				char str[ti.get_dimension() + 1];
				double v = (double)d.bigintval/pow(10, ti.get_scale());
				sprintf(str, "%*.*f", ti.get_dimension(), ti.get_scale(), v);
				return std::string(str);
			}
		case kINT:
			return std::to_string(d.intval);
		case kSMALLINT:
			return std::to_string(d.smallintval);
		case kBIGINT:
			return std::to_string(d.bigintval);
		case kFLOAT:
			return std::to_string(d.floatval);
		case kDOUBLE:
			return std::to_string(d.doubleval);
		case kTIME:
			{
				std::tm tm_struct;
				gmtime_r(&d.timeval, &tm_struct);
				char buf[9];
				strftime(buf, 9, "%T", &tm_struct);
				return std::string(buf);
			}
		case kTIMESTAMP:
			{
				std::tm tm_struct;
				gmtime_r(&d.timeval, &tm_struct);
				char buf[20];
				strftime(buf, 20, "%F %T", &tm_struct);
				return std::string(buf);
			}
		case kDATE:
			{
				std::tm tm_struct;
				gmtime_r(&d.timeval, &tm_struct);
				char buf[11];
				strftime(buf, 11, "%F", &tm_struct);
				return std::string(buf);
			}
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
