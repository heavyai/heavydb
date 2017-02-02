#ifndef QUERYENGINE_STRINGDICTIONARYGENERATION_H
#define QUERYENGINE_STRINGDICTIONARYGENERATION_H

#include <unordered_map>

class StringDictionaryGenerations {
 public:
  void setGeneration(const uint32_t id, const size_t generation);

  ssize_t getGeneration(const uint32_t id) const;

  const std::unordered_map<uint32_t, size_t>& asMap() const;

  void clear();

 private:
  std::unordered_map<uint32_t, size_t> id_to_generation_;
};

#endif  // QUERYENGINE_STRINGDICTIONARYGENERATION_H
