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

#pragma once

#include "StringViewSource.h"
#include "StringViewToArrayEncoder.h"
#include "StringViewToGeoEncoder.h"
#include "StringViewToScalarEncoder.h"
#include "StringViewToStringDictEncoder.h"
#include "StringViewToStringNoneEncoder.h"

namespace data_conversion {

struct ConversionFactoryParam {
  Chunk_NS::Chunk dst_chunk;
  Chunk_NS::Chunk scalar_temp_chunk;                             // used by array encoders
  std::list<Chunk_NS::Chunk> geo_chunks;                         // used by geo encoder
  std::list<std::unique_ptr<ChunkMetadata>> geo_chunk_metadata;  // used by geo encoder
  int db_id;
};

std::unique_ptr<BaseConvertEncoder> create_string_view_encoder(
    ConversionFactoryParam& param,
    const bool error_tracking_enabled,
    const bool geo_validate_geometry) {
  auto dst_type_info = param.geo_chunks.size()
                           ? param.geo_chunks.begin()->getColumnDesc()->columnType
                           : param.dst_chunk.getColumnDesc()->columnType;
  if (dst_type_info.is_dict_encoded_string()) {
    switch (dst_type_info.get_size()) {
      case 1:
        return std::make_unique<StringViewToStringDictEncoder<uint8_t>>(
            param.dst_chunk, error_tracking_enabled);
      case 2:
        return std::make_unique<StringViewToStringDictEncoder<uint16_t>>(
            param.dst_chunk, error_tracking_enabled);
      case 4:
        return std::make_unique<StringViewToStringDictEncoder<int32_t>>(
            param.dst_chunk, error_tracking_enabled);
      default:
        UNREACHABLE();
    }
  } else if (dst_type_info.is_none_encoded_string()) {
    return std::make_unique<StringViewToStringNoneEncoder>(param.dst_chunk,
                                                           error_tracking_enabled);
  } else if (dst_type_info.is_date_in_days()) {
    switch (dst_type_info.get_comp_param()) {
      case 0:
      case 32:
        return std::make_unique<StringViewToScalarEncoder<int32_t, int64_t>>(
            param.dst_chunk, error_tracking_enabled);
      case 16:
        return std::make_unique<StringViewToScalarEncoder<int16_t, int64_t>>(
            param.dst_chunk, error_tracking_enabled);
      default:
        UNREACHABLE();
    }
  } else if (dst_type_info.is_integer() || dst_type_info.is_boolean() ||
             dst_type_info.is_fp() || dst_type_info.is_decimal() ||
             dst_type_info.is_time_or_date()) {
    if (dst_type_info.get_compression() == kENCODING_NONE) {
      switch (dst_type_info.get_type()) {
        case kBOOLEAN:
        case kTINYINT:
          return std::make_unique<StringViewToScalarEncoder<int8_t>>(
              param.dst_chunk, error_tracking_enabled);
        case kSMALLINT:
          return std::make_unique<StringViewToScalarEncoder<int16_t>>(
              param.dst_chunk, error_tracking_enabled);
        case kINT:
          return std::make_unique<StringViewToScalarEncoder<int32_t>>(
              param.dst_chunk, error_tracking_enabled);
        case kBIGINT:
        case kNUMERIC:
        case kDECIMAL:
        case kTIME:
        case kTIMESTAMP:
        case kDATE:
          return std::make_unique<StringViewToScalarEncoder<int64_t>>(
              param.dst_chunk, error_tracking_enabled);
        case kFLOAT:
          return std::make_unique<StringViewToScalarEncoder<float>>(
              param.dst_chunk, error_tracking_enabled);
        case kDOUBLE:
          return std::make_unique<StringViewToScalarEncoder<double>>(
              param.dst_chunk, error_tracking_enabled);
        default:
          UNREACHABLE();
      }
    } else if (dst_type_info.get_compression() == kENCODING_FIXED) {
      switch (dst_type_info.get_type()) {
        case kSMALLINT: {
          switch (dst_type_info.get_comp_param()) {
            case 8:
              return std::make_unique<StringViewToScalarEncoder<int8_t, int16_t>>(
                  param.dst_chunk, error_tracking_enabled);
            default:
              UNREACHABLE();
          }
        } break;
        case kINT: {
          switch (dst_type_info.get_comp_param()) {
            case 8:
              return std::make_unique<StringViewToScalarEncoder<int8_t, int32_t>>(
                  param.dst_chunk, error_tracking_enabled);
            case 16:
              return std::make_unique<StringViewToScalarEncoder<int16_t, int32_t>>(
                  param.dst_chunk, error_tracking_enabled);
            default:
              UNREACHABLE();
          }
        } break;
        case kBIGINT:
        case kNUMERIC:
        case kDECIMAL: {
          switch (dst_type_info.get_comp_param()) {
            case 8:
              return std::make_unique<StringViewToScalarEncoder<int8_t, int64_t>>(
                  param.dst_chunk, error_tracking_enabled);
            case 16:
              return std::make_unique<StringViewToScalarEncoder<int16_t, int64_t>>(
                  param.dst_chunk, error_tracking_enabled);
            case 32:
              return std::make_unique<StringViewToScalarEncoder<int32_t, int64_t>>(
                  param.dst_chunk, error_tracking_enabled);
            default:
              UNREACHABLE();
          }
        } break;
        case kTIME:
        case kTIMESTAMP:
        case kDATE:
          return std::make_unique<StringViewToScalarEncoder<int32_t, int64_t>>(
              param.dst_chunk, error_tracking_enabled);
        default:
          UNREACHABLE();
      }
    } else {
      UNREACHABLE() << "unknown encoding type";
    }
  } else if (dst_type_info.is_array()) {
    auto dst_sub_type_info = dst_type_info.get_elem_type();
    if (dst_sub_type_info.is_dict_encoded_string()) {
      switch (dst_sub_type_info.get_size()) {
        case 4:
          return std::make_unique<
              StringViewToArrayEncoder<StringViewToStringDictEncoder<int32_t>>>(
              param.scalar_temp_chunk, param.dst_chunk, error_tracking_enabled);
        default:
          UNREACHABLE();
      }
    } else if (dst_sub_type_info.is_none_encoded_string()) {
      UNREACHABLE();
    } else if (dst_sub_type_info.is_date_in_days()) {
      switch (dst_sub_type_info.get_comp_param()) {
        case 0:
        case 32:
          return std::make_unique<
              StringViewToArrayEncoder<StringViewToScalarEncoder<int32_t, int64_t>>>(
              param.scalar_temp_chunk,
              param.dst_chunk,

              error_tracking_enabled);
        default:
          UNREACHABLE();
      }
    } else if (dst_sub_type_info.is_integer() || dst_sub_type_info.is_boolean() ||
               dst_sub_type_info.is_fp() || dst_sub_type_info.is_decimal() ||
               dst_sub_type_info.is_time_or_date()) {
      if (dst_sub_type_info.get_compression() == kENCODING_NONE) {
        switch (dst_sub_type_info.get_type()) {
          case kBOOLEAN:
            return std::make_unique<
                StringViewToArrayEncoder<StringViewToScalarEncoder<int8_t>>>(
                param.scalar_temp_chunk,
                param.dst_chunk,

                error_tracking_enabled);
          case kTINYINT:
            return std::make_unique<
                StringViewToArrayEncoder<StringViewToScalarEncoder<int8_t>>>(
                param.scalar_temp_chunk,
                param.dst_chunk,

                error_tracking_enabled);
          case kSMALLINT:
            return std::make_unique<
                StringViewToArrayEncoder<StringViewToScalarEncoder<int16_t>>>(
                param.scalar_temp_chunk,
                param.dst_chunk,

                error_tracking_enabled);
          case kINT:
            return std::make_unique<
                StringViewToArrayEncoder<StringViewToScalarEncoder<int32_t>>>(
                param.scalar_temp_chunk,
                param.dst_chunk,

                error_tracking_enabled);
          case kBIGINT:
          case kNUMERIC:
          case kDECIMAL:
          case kTIME:
          case kTIMESTAMP:
          case kDATE:
            return std::make_unique<
                StringViewToArrayEncoder<StringViewToScalarEncoder<int64_t>>>(
                param.scalar_temp_chunk,
                param.dst_chunk,

                error_tracking_enabled);
          case kFLOAT:
            return std::make_unique<
                StringViewToArrayEncoder<StringViewToScalarEncoder<float>>>(
                param.scalar_temp_chunk,
                param.dst_chunk,

                error_tracking_enabled);
          case kDOUBLE:
            return std::make_unique<
                StringViewToArrayEncoder<StringViewToScalarEncoder<double>>>(
                param.scalar_temp_chunk,
                param.dst_chunk,

                error_tracking_enabled);
          default:
            UNREACHABLE();
        }
      } else if (dst_sub_type_info.get_compression() == kENCODING_FIXED) {
        UNREACHABLE();
      } else {
        UNREACHABLE() << "unknown encoding type";
      }
    }
  } else if (dst_type_info.is_geometry()) {
    return std::make_unique<StringViewToGeoEncoder>(param.geo_chunks,
                                                    param.geo_chunk_metadata,
                                                    error_tracking_enabled,
                                                    geo_validate_geometry);
  }

  UNREACHABLE() << "could not find appropriate encoder to create, conversion use case is "
                   "unsupported";

  return {};
}

std::unique_ptr<BaseSource> create_source(const Chunk_NS::Chunk& input, const int db_id) {
  auto src_type_info = input.getColumnDesc()->columnType;
  CHECK(src_type_info.is_string()) << "Only string source types currently implemented.";

  if (src_type_info.is_dict_encoded_string() || src_type_info.is_none_encoded_string()) {
    return std::make_unique<StringViewSource>(input);
  } else {
    UNREACHABLE() << "unknown string type, not supported";
  }

  UNREACHABLE();
  return {};
}

}  // namespace data_conversion
