#ifndef STRINGDICTIONARY_STRINGDICTIONARYPROXY_H
#define STRINGDICTIONARY_STRINGDICTIONARYPROXY_H

#include "../Shared/mapd_shared_mutex.h"
#include "StringDictionary.h"
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <map>
#include <string>
#include <tuple>
#include <vector>

// used to access a StringDictionary when transient strings are involved
class StringDictionaryProxy {
 public:
  StringDictionaryProxy(std::shared_ptr<StringDictionary> sd) noexcept;

  int32_t getOrAdd(const std::string& str) noexcept;
  StringDictionary* getDictionary() noexcept;
  int32_t getOrAddTransient(const std::string& str) noexcept;
  int32_t get(const std::string& str) const noexcept;
  std::string getString(int32_t string_id) const noexcept;
  std::pair<char*, size_t> getStringBytes(int32_t string_id) const noexcept;
  size_t size() const noexcept;

  std::vector<std::string> getLike(const std::string& pattern,
                                   const bool icase,
                                   const bool is_simple,
                                   const char escape) const noexcept;

  std::vector<std::string> getRegexpLike(const std::string& pattern, const char escape) const noexcept;

 private:
  std::shared_ptr<StringDictionary> string_dict_;
  std::map<int32_t, std::string> transient_int_to_str_;
  std::map<std::string, int32_t> transient_str_to_int_;
  mutable mapd_shared_mutex rw_mutex_;
};
#endif  // STRINGDICTIONARY_STRINGDICTIONARYPROXY_H