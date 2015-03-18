/**
 * @file		StringLike.h
 * @author	Wei Hong <wei@mapd.com>
 * @brief		Functions to support the LIKE and ILIKE operator in SQL.  Only
 * single-byte character set is supported for now.
 * 
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/

#ifndef STRING_LIKE_H
#define STRING_LIKE_H

/*
 * @brief string_like performs the SQL LIKE and ILIKE operation
 * @param str string argument to be matched against pattern.  single-byte 
 * character set only for now. null-termination not required.
 * @param str_len length of str
 * @param pattern pattern string for SQL LIKE
 * @param pat_len length of pattern
 * @param escape_char the escape character.  '\\' is expected by default.
 * @param is_ilike true if it is ILIKE, i.e., case-insensitive matching
 * @return true if str matchs pattern, false otherwise.  error condition
 * not handled for now.
 */
extern "C" bool
string_like(const char *str, int str_len, const char *pattern, int pat_len, char escape_char, bool is_ilike);


#endif // STRING_LIKE_H
