#include "StringDictionary.h"
#include "StringDictionaryProxy.h"
#include "../Shared/sqltypes.h"
#include "../Utils/StringLike.h"
#include "../Utils/Regexp.h"
#include "Shared/thread_count.h"

#include <glog/logging.h>
#include <sys/fcntl.h>

#include <thread>

StringDictionaryProxy::StringDictionaryProxy(std::shared_ptr<StringDictionary> sd) noexcept : string_dict_(sd) {}

StringDictionaryProxy::~StringDictionaryProxy() noexcept {}

int32_t StringDictionaryProxy::getOrAddTransient(const std::string& str) noexcept {
  mapd_lock_guard<mapd_shared_mutex> write_lock(rw_mutex_);
  auto transient_id = string_dict_->get(str);
  if (transient_id != StringDictionary::INVALID_STR_ID) {
    return transient_id;
  }
  const auto it = transient_str_to_int_.find(str);
  if (it != transient_str_to_int_.end()) {
    return it->second;
  }
  transient_id = -(transient_str_to_int_.size() + 2);  // make sure it's not INVALID_STR_ID
  {
    auto it_ok = transient_str_to_int_.insert(std::make_pair(str, transient_id));
    CHECK(it_ok.second);
  }
  {
    auto it_ok = transient_int_to_str_.insert(std::make_pair(transient_id, str));
    CHECK(it_ok.second);
  }
  return transient_id;
}

int32_t StringDictionaryProxy::get(const std::string& str) const noexcept {
  mapd_shared_lock<mapd_shared_mutex> read_lock(rw_mutex_);
  auto str_id = string_dict_->get(str);
  if (str_id != StringDictionary::INVALID_STR_ID || transient_str_to_int_.empty()) {
    return str_id;
  }
  auto it = transient_str_to_int_.find(str);
  return it != transient_str_to_int_.end() ? it->second : StringDictionary::INVALID_STR_ID;
}

std::string StringDictionaryProxy::getString(int32_t string_id) const noexcept {
  mapd_shared_lock<mapd_shared_mutex> read_lock(rw_mutex_);
  if (string_id >= 0) {
    return string_dict_->getString(string_id);
  }
  CHECK_NE(StringDictionary::INVALID_STR_ID, string_id);
  auto it = transient_int_to_str_.find(string_id);
  CHECK(it != transient_int_to_str_.end());
  return it->second;
}

namespace {

bool is_like(const std::string& str,
             const std::string& pattern,
             const bool icase,
             const bool is_simple,
             const char escape) {
  return icase ? (is_simple ? string_ilike_simple(str.c_str(), str.size(), pattern.c_str(), pattern.size())
                            : string_ilike(str.c_str(), str.size(), pattern.c_str(), pattern.size(), escape))
               : (is_simple ? string_like_simple(str.c_str(), str.size(), pattern.c_str(), pattern.size())
                            : string_like(str.c_str(), str.size(), pattern.c_str(), pattern.size(), escape));
}

}  // namespace

std::vector<std::string> StringDictionaryProxy::getLike(const std::string& pattern,
                                                        const bool icase,
                                                        const bool is_simple,
                                                        const char escape) const noexcept {
  auto result = string_dict_->getLike(pattern, icase, is_simple, escape);
  for (const auto& kv : transient_int_to_str_) {
    const auto str = getString(kv.first);
    if (is_like(str, pattern, icase, is_simple, escape)) {
      result.push_back(str);
    }
  }
  return result;
}

namespace {

bool is_regexp_like(const std::string& str, const std::string& pattern, const char escape) {
  return regexp_like(str.c_str(), str.size(), pattern.c_str(), pattern.size(), escape);
}

}  // namespace

std::vector<std::string> StringDictionaryProxy::getRegexpLike(const std::string& pattern, const char escape) const
    noexcept {
  auto result = string_dict_->getRegexpLike(pattern, escape);
  for (const auto& kv : transient_int_to_str_) {
    const auto str = getString(kv.first);
    if (is_regexp_like(str, pattern, escape)) {
      result.push_back(str);
    }
  }
  return result;
}

void StringDictionaryProxy::clearTransient() noexcept {
  mapd_lock_guard<mapd_shared_mutex> write_lock(rw_mutex_);
  decltype(transient_int_to_str_)().swap(transient_int_to_str_);
  decltype(transient_str_to_int_)().swap(transient_str_to_int_);
}

int32_t StringDictionaryProxy::getOrAdd(const std::string& str) noexcept {
  return string_dict_->getOrAdd(str);
}

std::pair<char*, size_t> StringDictionaryProxy::getStringBytes(int32_t string_id) const noexcept {
  return string_dict_.get()->getStringBytes(string_id);
}

size_t StringDictionaryProxy::size() const noexcept {
  return string_dict_.get()->size();
}

StringDictionary* StringDictionaryProxy::getDictionary() noexcept {
  return string_dict_.get();
}