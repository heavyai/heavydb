#ifndef DICTREF_H
#define DICTREF_H

#include <cstdint>
#include <cstdlib>
#include <functional>

struct dict_ref_t {
  int32_t dbId;
  int32_t dictId;

  dict_ref_t() {}
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
    return (this->dbId < rhs.dbId) ? true : (this->dbId == rhs.dbId) ? this->dictId < rhs.dictId : false;
  }

  inline size_t operator()(const struct dict_ref_t& ref) const noexcept {
    std::hash<int32_t> int32_hash;
    return int32_hash(ref.dictId) ^ (int32_hash(ref.dbId) << 2);
  }
};

typedef struct dict_ref_t DictRef;

#endif
