#pragma once

//
// adapted and updated from example by https://github.com/facontidavide
// provides optimized find() and try_emplace() without std::string reallocation
//

#include <string>
#include <string_view>
#include <unordered_map>

template <typename ValueT>
class string_view_unordered_map : public std::unordered_map<std::string, ValueT> {
 public:
  using MapT = typename std::unordered_map<std::string, ValueT>;

  auto find(std::string_view key) const {
    tmp_key_.reserve(key.size());
    tmp_key_.assign(key.data(), key.size());
    return MapT::find(tmp_key_);
  }

  auto find(const std::string& key) const { return MapT::find(key); }

  auto find(const char* key) const {
    tmp_key_.assign(key);
    return MapT::find(tmp_key_);
  }

  template <typename... ArgsT>
  auto try_emplace(std::string_view key, ArgsT&&... args) {
    tmp_key_.reserve(key.size());
    tmp_key_.assign(key.data(), key.size());
    return MapT::try_emplace(tmp_key_, std::forward<ArgsT>(args)...);
  }

  template <typename... ArgsT>
  auto try_emplace(const std::string& key, ArgsT&&... args) {
    return MapT::try_emplace(key, std::forward<ArgsT>(args)...);
  }

  template <typename... ArgsT>
  auto try_emplace(const char* key, ArgsT&&... args) {
    tmp_key_.assign(key);
    return MapT::try_emplace(tmp_key_, std::forward<ArgsT>(args)...);
  }

 private:
  thread_local static std::string tmp_key_;
};

template <typename ValueT>
thread_local std::string string_view_unordered_map<ValueT>::tmp_key_;
