/*
 * Copyright 2017 MapD Technologies, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Encoder.h"
#include "NoneEncoder.h"
#include "FixedLengthEncoder.h"
#include "StringNoneEncoder.h"
#include "ArrayNoneEncoder.h"
#include <glog/logging.h>

Encoder* Encoder::Create(Data_Namespace::AbstractBuffer* buffer, const SQLTypeInfo sqlType) {
  switch (sqlType.get_compression()) {
    case kENCODING_NONE: {
      switch (sqlType.get_type()) {
        case kBOOLEAN: {
          return new NoneEncoder<int8_t>(buffer);
          break;
        }
        case kSMALLINT: {
          return new NoneEncoder<int16_t>(buffer);
          break;
        }
        case kINT: {
          return new NoneEncoder<int32_t>(buffer);
          break;
        }
        case kBIGINT:
        case kNUMERIC:
        case kDECIMAL: {
          return new NoneEncoder<int64_t>(buffer);
          break;
        }
        case kFLOAT: {
          return new NoneEncoder<float>(buffer);
          break;
        }
        case kDOUBLE: {
          return new NoneEncoder<double>(buffer);
          break;
        }
        case kTEXT:
        case kVARCHAR:
        case kCHAR:
          return new StringNoneEncoder(buffer);
        case kARRAY:
          return new ArrayNoneEncoder(buffer);
        case kTIME:
        case kTIMESTAMP:
        case kDATE:
          return new NoneEncoder<time_t>(buffer);
        default: { return 0; }
      }
      break;
    }
    case kENCODING_FIXED: {
      switch (sqlType.get_type()) {
        case kSMALLINT: {
          switch (sqlType.get_comp_param()) {
            case 8:
              return new FixedLengthEncoder<int16_t, int8_t>(buffer);
              break;
            case 16:
              return new NoneEncoder<int16_t>(buffer);
              break;
            default:
              return 0;
              break;
          }
          break;
        }
        case kINT: {
          switch (sqlType.get_comp_param()) {
            case 8:
              return new FixedLengthEncoder<int32_t, int8_t>(buffer);
              break;
            case 16:
              return new FixedLengthEncoder<int32_t, int16_t>(buffer);
              break;
            case 32:
              return new NoneEncoder<int32_t>(buffer);
              break;
            default:
              return 0;
              break;
          }
        } break;
        case kBIGINT:
        case kNUMERIC:
        case kDECIMAL: {
          switch (sqlType.get_comp_param()) {
            case 8:
              return new FixedLengthEncoder<int64_t, int8_t>(buffer);
              break;
            case 16:
              return new FixedLengthEncoder<int64_t, int16_t>(buffer);
              break;
            case 32:
              return new FixedLengthEncoder<int64_t, int32_t>(buffer);
              break;
            case 64:
              return new NoneEncoder<int64_t>(buffer);
              break;
            default:
              return 0;
              break;
          }
          break;
        }
        case kTIME:
        case kTIMESTAMP:
        case kDATE:
          return new FixedLengthEncoder<time_t, int32_t>(buffer);
          break;
        default: {
          return 0;
          break;
        }
      }  // switch (sqlType)
      break;
    }  // Case: kENCODING_FIXED
    case kENCODING_DICT: {
      if (sqlType.get_type() == kARRAY) {
        CHECK(IS_STRING(sqlType.get_subtype()));
        return new ArrayNoneEncoder(buffer);
      } else {
        CHECK(sqlType.is_string());
        switch (sqlType.get_size()) {
          case 1:
            return new NoneEncoder<uint8_t>(buffer);
            break;
          case 2:
            return new NoneEncoder<uint16_t>(buffer);
            break;
          case 4:
            return new NoneEncoder<int32_t>(buffer);
            break;
          default:
            CHECK(false);
            break;
        }
      }
      break;
    }
    default: {
      return 0;
      break;
    }
  }  // switch (encodingType)
  return 0;
}

void Encoder::getMetadata(ChunkMetadata& chunkMetadata) {
  // chunkMetadata = metadataTemplate_; // invoke copy constructor
  chunkMetadata.sqlType = buffer_->sqlType;
  chunkMetadata.numBytes = buffer_->size();
  chunkMetadata.numElements = numElems;
}

ChunkMetadata Encoder::getMetadata(const SQLTypeInfo& ti) {
  CHECK(false);
  return {};
}

void Encoder::updateStats(const int64_t, const bool) {
  CHECK(false);
}

void Encoder::updateStats(const double, const bool) {
  CHECK(false);
}

void Encoder::reduceStats(const Encoder&) {
  CHECK(false);
}
