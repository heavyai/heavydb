/**
 * @file		StringNoneEncoder.h
 * @author	Wei Hong <wei@map-d.com>
 * @brief		For unencoded strings
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/
#ifndef STRING_NONE_ENCODER_H
#define STRING_NONE_ENCODER_H

#include <vector>
#include <string>
#include <cassert>
#include "AbstractBuffer.h"
#include "ChunkMetadata.h"
#include "Encoder.h"

using Data_Namespace::AbstractBuffer;

class StringNoneEncoder : public Encoder {
 public:
  StringNoneEncoder(AbstractBuffer* buffer) : Encoder(buffer), index_buf(nullptr), last_offset(-1), has_nulls(false) {}

  size_t getNumElemsForBytesInsertData(const std::vector<std::string>* srcData,
                                       const int start_idx,
                                       const size_t numAppendElems,
                                       const size_t byteLimit);

  ChunkMetadata appendData(int8_t*& srcData, const size_t numAppendElems) {
    assert(false);  // should never be called for strings
    ChunkMetadata chunkMetadata;
    getMetadata(chunkMetadata);
    return chunkMetadata;
  }

  ChunkMetadata appendData(const std::vector<std::string>* srcData, const int start_idx, const size_t numAppendElems);

  void getMetadata(ChunkMetadata& chunkMetadata) {
    Encoder::getMetadata(chunkMetadata);  // call on parent class
    chunkMetadata.chunkStats.min.stringval = nullptr;
    chunkMetadata.chunkStats.max.stringval = nullptr;
    chunkMetadata.chunkStats.has_nulls = has_nulls;
  }

  void writeMetadata(FILE* f) {
    // assumes pointer is already in right place
    fwrite((int8_t*)&numElems, sizeof(size_t), 1, f);
    fwrite((int8_t*)&has_nulls, sizeof(bool), 1, f);
  }

  void readMetadata(FILE* f) {
    // assumes pointer is already in right place
    CHECK_NE(fread((int8_t*)&numElems, sizeof(size_t), size_t(1), f), size_t(0));
    CHECK_NE(fread((int8_t*)&has_nulls, sizeof(bool), size_t(1), f), size_t(0));
  }

  void copyMetadata(const Encoder* copyFromEncoder) {
    numElems = copyFromEncoder->numElems;
    has_nulls = static_cast<const StringNoneEncoder*>(copyFromEncoder)->has_nulls;
  }

  AbstractBuffer* get_index_buf() const { return index_buf; }
  void set_index_buf(AbstractBuffer* buf) { index_buf = buf; }

 private:
  AbstractBuffer* index_buf;
  StringOffsetT last_offset;
  bool has_nulls;

};  // class StringNoneEncoder

#endif  // STRING_NONE_ENCODER_H
