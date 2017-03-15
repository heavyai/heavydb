#ifndef QUERYENGINE_TABLEGENERATIONS_H
#define QUERYENGINE_TABLEGENERATIONS_H

#include <unordered_map>

struct TableGeneration {
  const size_t tuple_count;
  const size_t start_rowid;
};

class TableGenerations {
 public:
  void setGeneration(const uint32_t id, const TableGeneration& generation);

  const TableGeneration& getGeneration(const uint32_t id) const;

  const std::unordered_map<uint32_t, TableGeneration>& asMap() const;

  void clear();

 private:
  std::unordered_map<uint32_t, TableGeneration> id_to_generation_;
};

#endif  // QUERYENGINE_TABLEGENERATIONS_H
