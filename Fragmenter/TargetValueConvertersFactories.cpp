/*
 * Copyright 2018, OmniSci, Inc.
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

#include "TargetValueConvertersFactories.h"
#include "TargetValueConvertersImpl.h"

template <typename SOURCE_TYPE, typename TARGET_TYPE>
struct NumericConverterFactory {
  using ConverterType = NumericValueConverter<SOURCE_TYPE, TARGET_TYPE>;

  std::unique_ptr<ConverterType> create(ConverterCreateParameter param) {
    SOURCE_TYPE source_null_value =
        static_cast<SOURCE_TYPE>(inline_int_null_value<SOURCE_TYPE>());

    switch (param.type.get_size()) {
      case 8:
        source_null_value = static_cast<SOURCE_TYPE>(inline_int_null_value<int64_t>());
        break;
      case 4:
        source_null_value = static_cast<SOURCE_TYPE>(inline_int_null_value<int32_t>());
        break;
      case 2:
        source_null_value = static_cast<SOURCE_TYPE>(inline_int_null_value<int16_t>());
        break;
      case 1:
        source_null_value = static_cast<SOURCE_TYPE>(inline_int_null_value<int8_t>());
        break;
      default:
        CHECK(false);
    }

    TARGET_TYPE target_null_value =
        static_cast<TARGET_TYPE>(inline_int_null_value<TARGET_TYPE>());

    return std::make_unique<NumericValueConverter<SOURCE_TYPE, TARGET_TYPE>>(
        param.target,
        param.num_rows,
        source_null_value,
        target_null_value,
        param.can_be_null);
  }

  std::unique_ptr<TargetValueConverter> operator()(ConverterCreateParameter param) {
    return create(param);
  }
};

template <>
std::unique_ptr<NumericValueConverter<double, double>>
NumericConverterFactory<double, double>::create(ConverterCreateParameter param) {
  double null_value = inline_fp_null_value<double>();
  return std::make_unique<NumericValueConverter<double, double>>(
      param.target, param.num_rows, null_value, null_value, param.can_be_null);
}

template <>
std::unique_ptr<NumericValueConverter<float, float>>
NumericConverterFactory<float, float>::create(ConverterCreateParameter param) {
  float null_value = inline_fp_null_value<float>();
  return std::make_unique<NumericValueConverter<float, float>>(
      param.target, param.num_rows, null_value, null_value, param.can_be_null);
}

template <typename TARGET_TYPE>
struct DictionaryConverterFactory {
  using ConverterType = DictionaryValueConverter<TARGET_TYPE>;

  std::unique_ptr<ConverterType> create(ConverterCreateParameter param) {
    TARGET_TYPE target_null_value = inline_int_null_value<int32_t>();

    switch (param.type.get_size()) {
      case 4:
        target_null_value = static_cast<TARGET_TYPE>(inline_int_null_value<int32_t>());
        break;
      case 2:
        target_null_value = static_cast<TARGET_TYPE>(inline_int_null_value<uint16_t>());
        break;
      case 1:
        target_null_value = static_cast<TARGET_TYPE>(inline_int_null_value<uint8_t>());
        break;
      default:
        CHECK(false);
    }

    return std::make_unique<DictionaryValueConverter<TARGET_TYPE>>(
        param.cat,
        param.source,
        param.target,
        param.num_rows,
        target_null_value,
        NULL_INT,
        param.can_be_null,
        param.always_expect_strings);
  }

  std::unique_ptr<TargetValueConverter> operator()(ConverterCreateParameter param) {
    return create(param);
  }
};

struct TextConverterFactory {
  std::unique_ptr<TargetValueConverter> operator()(ConverterCreateParameter param) {
    if (param.target->columnType.get_compression() == kENCODING_NONE) {
      return std::make_unique<StringValueConverter>(param.target, param.num_rows);
    } else if (param.target->columnType.get_compression() == kENCODING_DICT) {
      auto size = param.target->columnType.get_size();
      if (4 == size) {
        DictionaryConverterFactory<int32_t> factory;
        return factory.create(param);
      } else if (2 == size) {
        DictionaryConverterFactory<uint16_t> factory;
        return factory.create(param);
      } else if (1 == size) {
        DictionaryConverterFactory<uint8_t> factory;
        return factory.create(param);
      }
    }

    throw std::runtime_error("Unsupported text column type");
  }
};

struct DateConverterFactory {
  std::unique_ptr<TargetValueConverter> operator()(ConverterCreateParameter param) {
    if (param.target->columnType.is_date_in_days() && param.type.get_size() == 4) {
      return std::make_unique<DateValueConverter>(
          param.target,
          param.num_rows,
          static_cast<int64_t>(inline_int_null_value<int32_t>()),
          NULL_BIGINT,
          param.can_be_null);
    } else if (param.target->columnType.is_date_in_days() && param.type.get_size() == 2) {
      return std::make_unique<DateValueConverter>(
          param.target,
          param.num_rows,
          static_cast<int64_t>(inline_int_null_value<int16_t>()),
          NULL_BIGINT,
          param.can_be_null);
    } else {
      NumericConverterFactory<int64_t, int64_t> factory;
      return factory.create(param);
    }
  }
};

template <typename ELEMENT_FACTORY>
struct ArrayConverterFactory {
  ELEMENT_FACTORY element_factory_;

  std::unique_ptr<ArrayValueConverter<typename ELEMENT_FACTORY::ConverterType>> create(
      ConverterCreateParameter param) {
    auto elem_type = param.target->columnType.get_elem_type();
    ConverterCreateParameter elementConverterFactoryParam{0,
                                                          param.cat,
                                                          param.source,
                                                          param.target,
                                                          elem_type,
                                                          false,
                                                          param.always_expect_strings};

    auto elementConverter = element_factory_.create(elementConverterFactoryParam);
    return std::make_unique<ArrayValueConverter<typename ELEMENT_FACTORY::ConverterType>>(
        param.target, param.num_rows, std::move(elementConverter), param.can_be_null);
  }

  std::unique_ptr<TargetValueConverter> operator()(ConverterCreateParameter param) {
    return create(param);
  }
};

struct ArraysConverterFactory {
  std::unique_ptr<TargetValueConverter> operator()(ConverterCreateParameter param) {
    const static std::map<std::tuple<SQLTypes, EncodingType>,
                          std::function<std::unique_ptr<TargetValueConverter>(
                              ConverterCreateParameter param)>>
        array_converter_factories{
            {{kBIGINT, kENCODING_NONE},
             ArrayConverterFactory<NumericConverterFactory<int64_t, int64_t>>()},
            {{kINT, kENCODING_NONE},
             ArrayConverterFactory<NumericConverterFactory<int64_t, int32_t>>()},
            {{kSMALLINT, kENCODING_NONE},
             ArrayConverterFactory<NumericConverterFactory<int64_t, int16_t>>()},
            {{kTINYINT, kENCODING_NONE},
             ArrayConverterFactory<NumericConverterFactory<int64_t, int8_t>>()},
            {{kDECIMAL, kENCODING_NONE},
             ArrayConverterFactory<NumericConverterFactory<int64_t, int64_t>>()},
            {{kNUMERIC, kENCODING_NONE},
             ArrayConverterFactory<NumericConverterFactory<int64_t, int64_t>>()},
            {{kTIMESTAMP, kENCODING_NONE},
             ArrayConverterFactory<NumericConverterFactory<int64_t, int64_t>>()},
            {{kDATE, kENCODING_NONE},
             ArrayConverterFactory<NumericConverterFactory<int64_t, int64_t>>()},
            {{kTIME, kENCODING_NONE},
             ArrayConverterFactory<NumericConverterFactory<int64_t, int64_t>>()},
            {{kBOOLEAN, kENCODING_NONE},
             ArrayConverterFactory<NumericConverterFactory<int64_t, int8_t>>()},
            {{kDOUBLE, kENCODING_NONE},
             ArrayConverterFactory<NumericConverterFactory<double, double>>()},
            {{kFLOAT, kENCODING_NONE},
             ArrayConverterFactory<NumericConverterFactory<float, float>>()},
            {{kTEXT, kENCODING_DICT},
             ArrayConverterFactory<DictionaryConverterFactory<int32_t>>()},
            {{kCHAR, kENCODING_DICT},
             ArrayConverterFactory<DictionaryConverterFactory<int32_t>>()},
            {{kVARCHAR, kENCODING_DICT},
             ArrayConverterFactory<DictionaryConverterFactory<int32_t>>()}};

    auto elem_type = param.target->columnType.get_elem_type();
    auto factory = array_converter_factories.find(
        {elem_type.get_type(), elem_type.get_compression()});

    if (factory != array_converter_factories.end()) {
      return factory->second(param);
    }

    throw std::runtime_error("Unsupported array column type");
  }
};

template <typename CONVERTER>
struct GeoConverterFactory {
  std::unique_ptr<TargetValueConverter> operator()(ConverterCreateParameter param) {
    return std::make_unique<CONVERTER>(param.cat, param.num_rows, param.target);
  }
};

std::unique_ptr<TargetValueConverter> TargetValueConverterFactory::create(
    ConverterCreateParameter param) {
  static const std::map<SQLTypes,
                        std::function<std::unique_ptr<TargetValueConverter>(
                            ConverterCreateParameter param)>>
      factories{{kBIGINT, NumericConverterFactory<int64_t, int64_t>()},
                {kINT, NumericConverterFactory<int64_t, int32_t>()},
                {kSMALLINT, NumericConverterFactory<int64_t, int16_t>()},
                {kTINYINT, NumericConverterFactory<int64_t, int8_t>()},
                {kDECIMAL, NumericConverterFactory<int64_t, int64_t>()},
                {kNUMERIC, NumericConverterFactory<int64_t, int64_t>()},
                {kTIMESTAMP, NumericConverterFactory<int64_t, int64_t>()},
                {kDATE, DateConverterFactory()},
                {kTIME, NumericConverterFactory<int64_t, int64_t>()},
                {kBOOLEAN, NumericConverterFactory<int64_t, int8_t>()},
                {kDOUBLE, NumericConverterFactory<double, double>()},
                {kFLOAT, NumericConverterFactory<float, float>()},
                {kTEXT, TextConverterFactory()},
                {kCHAR, TextConverterFactory()},
                {kVARCHAR, TextConverterFactory()},
                {kARRAY, ArraysConverterFactory()},
                {kPOINT, GeoConverterFactory<GeoPointValueConverter>()},
                {kLINESTRING, GeoConverterFactory<GeoLinestringValueConverter>()},
                {kPOLYGON, GeoConverterFactory<GeoPolygonValueConverter>()},
                {kMULTIPOLYGON, GeoConverterFactory<GeoMultiPolygonValueConverter>()}};

  auto factory = factories.find(param.target->columnType.get_type());

  if (factory != factories.end()) {
    return factory->second(param);
  } else {
    throw std::runtime_error("Unsupported column type: " +
                             param.target->columnType.get_type_name());
  }
}
