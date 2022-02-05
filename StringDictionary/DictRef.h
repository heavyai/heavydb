#ifndef DICTREF_H
#define DICTREF_H

#include <cstdint>
#include <cstdlib>
#include <functional>

struct dict_ref_t {
  int32_t dbId;
  int32_t dictId;

  static constexpr int32_t invalidDbId{-1};
  static constexpr int32_t invalidDictId{-1};
  static constexpr int32_t literalsDictId{0};

  dict_ref_t() : dbId(invalidDbId), dictId(invalidDictId) {}
  dict_ref_t(int32_t db_id, int32_t dict_id) : dbId(db_id), dictId(dict_id) {}

  inline bool operator==(const struct dict_ref_t& rhs) const {
    return this->dictId == rhs.dictId && this->dbId == rhs.dbId;
  }

  inline struct dict_ref_t& operator=(const struct dict_ref_t& rhs) {
    this->dbId = rhs.dbId;
    this->dictId = rhs.dictId;
    return *this;
  };

  inline bool operator<(const struct dict_ref_t& rhs) const {
    return (this->dbId < rhs.dbId)
               ? true
               : (this->dbId == rhs.dbId) ? this->dictId < rhs.dictId : false;
  }

  inline size_t operator()(const struct dict_ref_t& ref) const noexcept {
    std::hash<int32_t> int32_hash;
    return int32_hash(ref.dictId) ^ (int32_hash(ref.dbId) << 2);
  }

  inline std::string toString() const {
    return "(db_id: " + std::to_string(dbId) + ", dict_id: " + std::to_string(dictId) +
           ")";
  }

  static dict_ref_t InvalidDictRef() { return dict_ref_t(); }
};

using DictRef = struct dict_ref_t;

#endif
