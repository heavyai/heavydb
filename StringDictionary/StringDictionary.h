#ifndef STRINGDICTIONARY_STRINGDICTIONARY_H
#define STRINGDICTIONARY_STRINGDICTIONARY_H

#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <string>
#include <vector>


class StringDictionary {
public:
  StringDictionary(const std::string& folder,
                   const bool recover = true,
                   size_t initial_capacity = 256);
  ~StringDictionary();

  int32_t getOrAdd(const std::string& str);
  int32_t get(const std::string& str) const;
  std::string getString(int32_t string_id) const;

private:
  struct StringIdxEntry {
    uint64_t off  : 48;
    uint64_t size : 16;
  };

  bool fillRateIsHigh() const;
  void increaseCapacity();
  int32_t getOrAddImpl(const std::string& str, bool recover);
  std::string getStringChecked(const int string_id) const;
  int32_t computeBucket(const std::string& str, const std::vector<int32_t>& data) const;
  void appendToStorage(const std::string& str);
  std::pair<std::string, bool> getStringFromStorage(const int string_id) const;
  void addPayloadCapacity();
  void addOffsetCapacity();
  size_t addStorageCapacity(int fd);

  const std::string folder_;
  size_t str_count_;
  std::vector<int32_t> str_ids_;
  std::string offsets_path_;
  int payload_fd_;
  int offset_fd_;
  StringIdxEntry* offset_map_;
  char* payload_map_;
  size_t offset_file_size_;
  size_t payload_file_size_;
  size_t payload_file_off_;

  static char* CANARY_BUFFER;
};

#endif  // STRINGDICTIONARY_STRINGDICTIONARY_H
