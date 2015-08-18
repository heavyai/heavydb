#ifndef NONE_ENCODER_H
#define NONE_ENCODER_H

#include "AbstractBuffer.h"
#include "Encoder.h"

template <typename T>
class NoneEncoder : public Encoder {
 public:
  NoneEncoder(Data_Namespace::AbstractBuffer* buffer)
      : Encoder(buffer),
        dataMin(std::numeric_limits<T>::max()),
        dataMax(std::numeric_limits<T>::min()),
        has_nulls(false) {}

  ChunkMetadata appendData(int8_t*& srcData, const size_t numAppendElems) {
    T* unencodedData = reinterpret_cast<T*>(srcData);
    for (size_t i = 0; i < numAppendElems; ++i) {
      T data = unencodedData[i];
      if (data == std::numeric_limits<T>::min())
        has_nulls = true;
      else {
        dataMin = std::min(dataMin, data);
        dataMax = std::max(dataMax, data);
      }
    }
    numElems += numAppendElems;
    buffer_->append(srcData, numAppendElems * sizeof(T));
    ChunkMetadata chunkMetadata;
    getMetadata(chunkMetadata);
    srcData += numAppendElems * sizeof(T);
    return chunkMetadata;
  }

  void getMetadata(ChunkMetadata& chunkMetadata) {
    Encoder::getMetadata(chunkMetadata);  // call on parent class
    chunkMetadata.fillChunkStats(dataMin, dataMax, has_nulls);
  }

  void writeMetadata(FILE* f) {
    // assumes pointer is already in right place
    CHECK_RET(fwrite((int8_t*)&numElems, sizeof(size_t), 1, f));
    CHECK_RET(fwrite((int8_t*)&dataMin, sizeof(T), 1, f));
    CHECK_RET(fwrite((int8_t*)&dataMax, sizeof(T), 1, f));
    CHECK_RET(fwrite((int8_t*)&has_nulls, sizeof(bool), 1, f));
  }

  void readMetadata(FILE* f) {
    // assumes pointer is already in right place
    CHECK_RET(fread((int8_t*)&numElems, sizeof(size_t), 1, f));
    CHECK_RET(fread((int8_t*)&dataMin, sizeof(T), 1, f));
    CHECK_RET(fread((int8_t*)&dataMax, sizeof(T), 1, f));
    CHECK_RET(fread((int8_t*)&has_nulls, sizeof(bool), 1, f));
  }

  void copyMetadata(const Encoder* copyFromEncoder) {
    numElems = copyFromEncoder->numElems;
    auto castedEncoder = reinterpret_cast<const NoneEncoder<T>*>(copyFromEncoder);
    dataMin = castedEncoder->dataMin;
    dataMax = castedEncoder->dataMax;
    has_nulls = castedEncoder->has_nulls;
  }

  T dataMin;
  T dataMax;
  bool has_nulls;

};  // class NoneEncoder

#endif  // NONE_ENCODER_H
