/*
 * Copyright 2022 HEAVY.AI, Inc.
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
#include "ArrayNoneEncoder.h"
#include "DateDaysEncoder.h"
#include "FixedLengthArrayNoneEncoder.h"
#include "FixedLengthEncoder.h"
#include "Logger/Logger.h"
#include "NoneEncoder.h"
#include "StringNoneEncoder.h"

Encoder* Encoder::Create(Data_Namespace::AbstractBuffer* buffer,
                         const SQLTypeInfo sqlType) {
  switch (sqlType.get_compression()) {
    case kENCODING_NONE: {
      switch (sqlType.get_type()) {
        case kBOOLEAN: {
          return new NoneEncoder<int8_t>(buffer);
          break;
        }
        case kTINYINT: {
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
        case kARRAY: {
          if (sqlType.get_size() > 0) {
            return new FixedLengthArrayNoneEncoder(buffer, sqlType.get_size());
          }
          return new ArrayNoneEncoder(buffer);
        }
        case kTIME:
        case kTIMESTAMP:
        case kDATE:
          return new NoneEncoder<int64_t>(buffer);
        case kPOINT:
        case kMULTIPOINT:
        case kLINESTRING:
        case kMULTILINESTRING:
        case kPOLYGON:
        case kMULTIPOLYGON:
          return new StringNoneEncoder(buffer);
        default: {
          return 0;
        }
      }
      break;
    }
    case kENCODING_DATE_IN_DAYS: {
      switch (sqlType.get_type()) {
        case kDATE:
          switch (sqlType.get_comp_param()) {
            case 0:
            case 32:
              return new DateDaysEncoder<int64_t, int32_t>(buffer);
              break;
            case 16:
              return new DateDaysEncoder<int64_t, int16_t>(buffer);
              break;
            default:
              return 0;
              break;
          }
          break;
        default:
          return 0;
          break;
      }
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
          break;
        }
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
          return new FixedLengthEncoder<int64_t, int32_t>(buffer);
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
        if (sqlType.get_size() > 0) {
          return new FixedLengthArrayNoneEncoder(buffer, sqlType.get_size());
        }
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
    case kENCODING_GEOINT: {
      switch (sqlType.get_type()) {
        case kPOINT:
        case kMULTIPOINT:
        case kLINESTRING:
        case kMULTILINESTRING:
        case kPOLYGON:
        case kMULTIPOLYGON:
          return new StringNoneEncoder(buffer);
        default: {
          return 0;
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

Encoder::Encoder(Data_Namespace::AbstractBuffer* buffer)
    : num_elems_(0)
    , buffer_(buffer)
    , decimal_overflow_validator_(buffer ? buffer->getSqlType() : SQLTypeInfo())
    , date_days_overflow_validator_(buffer ? buffer->getSqlType() : SQLTypeInfo()){};

ChunkMetadata Encoder::getMetadata() const {
  CHECK(buffer_);
  return ChunkMetadata(
      buffer_->getSqlType(), buffer_->size(), num_elems_, getChunkStats(), raster_tile_);
}

ChunkMetadata Encoder::getMetadata(const SQLTypeInfo& ti) {
  return ChunkMetadata(ti, 0, 0, synthesizeChunkStats(ti), raster_tile_);
}

void Encoder::readMetadata(FILE* f, int32_t version) {
  // assumes pointer is already in right place
  CHECK_EQ(fread(reinterpret_cast<int8_t*>(&num_elems_), sizeof(size_t), size_t(1), f),
           1U);
  readChunkStats(f);
  if (version >= MetadataVersion::kRaster) {
    auto& [width, height, local_coords] = raster_tile_;
    auto& [file_id, x, y] = local_coords;
    CHECK_EQ(fread(reinterpret_cast<int8_t*>(&width), sizeof(width), 1, f), 1U);
    CHECK_EQ(fread(reinterpret_cast<int8_t*>(&height), sizeof(height), 1, f), 1U);
    CHECK_EQ(fread(reinterpret_cast<int8_t*>(&file_id), sizeof(file_id), 1, f), 1U);
    CHECK_EQ(fread(reinterpret_cast<int8_t*>(&x), sizeof(x), 1, f), 1U);
    CHECK_EQ(fread(reinterpret_cast<int8_t*>(&y), sizeof(y), 1, f), 1U);
  }
}

void Encoder::writeMetadata(FILE* f) {
  // assumes pointer is already in right place
  CHECK_EQ(fwrite(reinterpret_cast<int8_t*>(&num_elems_), sizeof(size_t), 1, f), 1U);
  writeChunkStats(f);
  const auto& [width, height, local_coords] = raster_tile_;
  const auto& [file_id, x, y] = local_coords;
  CHECK_EQ(fwrite(reinterpret_cast<const int8_t*>(&width), sizeof(width), 1, f), 1U);
  CHECK_EQ(fwrite(reinterpret_cast<const int8_t*>(&height), sizeof(height), 1, f), 1U);
  CHECK_EQ(fwrite(reinterpret_cast<const int8_t*>(&file_id), sizeof(file_id), 1, f), 1U);
  CHECK_EQ(fwrite(reinterpret_cast<const int8_t*>(&x), sizeof(x), 1, f), 1U);
  CHECK_EQ(fwrite(reinterpret_cast<const int8_t*>(&y), sizeof(y), 1, f), 1U);
}

void Encoder::copyMetadata(const Encoder* copy_from_encoder) {
  num_elems_ = copy_from_encoder->getNumElems();
  raster_tile_ = copy_from_encoder->getRasterTileInfo();
  copyChunkStats(copy_from_encoder);
}

void Encoder::setMetadata(const ChunkMetadata& meta) {
  setNumElems(meta.numElements);
  setRasterTileInfo(meta.rasterTile);
  setChunkStats(meta.chunkStats);
}
