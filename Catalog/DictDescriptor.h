#ifndef DICT_DESCRIPTOR_H
#define DICT_DESCRIPTOR_H

#include <cassert>
#include <string>
#include "../Shared/sqltypes.h"
#include "../StringDictionary/StringDictionary.h"

/**
 * @type DictDescriptor
 * @brief Descriptor for a dictionary for a string columne
 *
 */

struct DictDescriptor {
  int dictId;
  std::string dictName;
  int dictNBits;
  bool dictIsShared;
  std::string dictFolderPath;
  StringDictionary* stringDict;
  DictDescriptor(int id, const std::string& name, int nbits, bool shared, std::string& fname)
      : dictId(id),
        dictName(name),
        dictNBits(nbits),
        dictIsShared(shared),
        dictFolderPath(fname),
        stringDict(nullptr) {}
};

#endif  // DICT_DESCRIPTOR
