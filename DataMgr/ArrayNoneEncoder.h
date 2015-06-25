/**
 * @file		ArrayNoneEncoder.h
 * @author	Wei Hong <wei@mapd.com>
 * @brief		unencoded array encoder
 * 
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/
#ifndef ARRAY_NONE_ENCODER_H
#define ARRAY_NONE_ENCODER_H

#include <cstring>
#include <memory>
#include <vector>
#include <string>
#include <cassert>
#include "AbstractBuffer.h"
#include "ChunkMetadata.h"
#include "Encoder.h"

using Data_Namespace::AbstractBuffer;

class ArrayNoneEncoder : public Encoder {

    public:
        ArrayNoneEncoder(AbstractBuffer *buffer): Encoder(buffer), index_buf(nullptr), last_offset(-1) {}

        ChunkMetadata appendData(int8_t * &srcData, const size_t numAppendElems) {
						assert(false); // should never be called for strings
            ChunkMetadata chunkMetadata;
            getMetadata(chunkMetadata);
            return chunkMetadata;
        }

				ChunkMetadata appendData(const std::vector<ArrayDatum> *srcData, const int start_idx, const size_t numAppendElems)
        {
          assert(index_buf != nullptr); // index_buf must be set before this.
          size_t index_size = numAppendElems * sizeof(StringOffsetT);
          if (numElems == 0)
            index_size += sizeof(StringOffsetT); // plus one for the initial offset of 0.
          index_buf->reserve(index_size);
          StringOffsetT offset = 0;
          if (numElems == 0) {
            index_buf->append((int8_t*)&offset, sizeof(StringOffsetT));  // write the inital 0 offset
            last_offset = 0;
          } else {
            if (last_offset < 0) {
              // need to read the last offset from buffer/disk
              index_buf->read((int8_t*)&last_offset, sizeof(StringOffsetT), index_buf->size() - sizeof(StringOffsetT), Data_Namespace::CPU_LEVEL);
              assert(last_offset >= 0);
            }
          }
          size_t data_size = 0;
          for (size_t n = start_idx; n < start_idx + numAppendElems; n++) {
            size_t len = (*srcData)[n].length;
            data_size += len;
          }
          buffer_->reserve(data_size);

          size_t inbuf_size = std::min(std::max(index_size, data_size), (size_t)MAX_INPUT_BUF_SIZE);
          int8_t *inbuf = (int8_t*)malloc(inbuf_size);
          std::unique_ptr<int8_t> gc_inbuf(inbuf);
          for (size_t num_appended = 0; num_appended < numAppendElems; ) {
            StringOffsetT *p = (StringOffsetT*)inbuf;
            size_t i;
            for (i = 0; num_appended < numAppendElems && i < inbuf_size/sizeof(StringOffsetT); i++, num_appended++) {
              p[i] = last_offset + (*srcData)[num_appended + start_idx].length;
              last_offset = p[i];
            }
            index_buf->append(inbuf, i * sizeof(StringOffsetT));
          }

          for (size_t num_appended = 0; num_appended < numAppendElems; ) {
            size_t size = 0;
            for (int i = start_idx + num_appended; num_appended < numAppendElems && size < inbuf_size;  i++, num_appended++) {
              size_t len = (*srcData)[i].length;
              if (len > inbuf_size) {
                // for large strings, append on its own
                if (size > 0)
                  buffer_->append(inbuf, size);
                size = 0;
                buffer_->append((*srcData)[i].data_ptr.get(), len);
                num_appended++;
                break;
              } else if (size + len > inbuf_size)
                break;
              char *dest = (char*)inbuf + size;
              if (len > 0)
                std::memcpy((void*)dest, (void*)(*srcData)[i].data_ptr.get(), len);
              size += len;
            }
            if (size > 0)
              buffer_->append(inbuf, size);
          }
          // make sure buffer_ is flushed even if no new data is appended to it 
          // (e.g. empty strings) because the metadata needs to be flushed.
          if (!buffer_->isDirty())
            buffer_->setDirty();

          numElems += numAppendElems;
          ChunkMetadata chunkMetadata;
          getMetadata(chunkMetadata);
          return chunkMetadata;
        }

        void getMetadata(ChunkMetadata &chunkMetadata) {
            Encoder::getMetadata(chunkMetadata); // call on parent class
            // @TODO(wei) keep min max of array elements
            chunkMetadata.chunkStats.min.arrayval = nullptr;
            chunkMetadata.chunkStats.max.arrayval = nullptr;
        }

        void writeMetadata(FILE *f) {
            // assumes pointer is already in right place
            CHECK_RET(fwrite((int8_t *)&numElems,sizeof(size_t),1,f));
        }

        void readMetadata(FILE *f) {
            // assumes pointer is already in right place
            CHECK_RET(fread((int8_t *)&numElems,sizeof(size_t),1,f));
        }

        void copyMetadata(const Encoder * copyFromEncoder) {
            numElems = copyFromEncoder -> numElems;
        }

				AbstractBuffer *get_index_buf() const { return index_buf; }
				void set_index_buf(AbstractBuffer *buf) { index_buf = buf; }
		private:
			AbstractBuffer *index_buf;
			StringOffsetT last_offset;

}; // class ArrayNoneEncoder

#endif // ARRAY_NONE_ENCODER_H
