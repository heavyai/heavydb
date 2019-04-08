/*
 * Copyright 2018 MapD Technologies, Inc.
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

#ifndef EXPERIMENTALTYPEUTILITIES_H
#define EXPERIMENTALTYPEUTILITIES_H
#ifndef __CUDACC__

#include <boost/optional.hpp>
#include <iostream>
#include <tuple>
#include <utility>

#include "ConstExprLib.h"
#include "sqltypes.h"

namespace Experimental {

struct UncapturedMetaType {};
struct UncapturedMetaTypeClass {};

enum MetaTypeClassifications { Geometry, Array, String };

template <typename T>
struct MetaSwitchTraits {
  static auto getType(T const& t) { return t; }
};

template <MetaTypeClassifications T>
struct MetaTypeClassDeterminant {
  template <typename SQL_TYPE_INFO>
  static auto isTargetClass(SQL_TYPE_INFO const& s) {
    return false;
  }
};

template <>
struct MetaTypeClassDeterminant<Geometry> {
  template <typename SQL_TYPE_INFO>
  static auto isTargetClass(SQL_TYPE_INFO const& s) {
    return s.is_geometry();
  }
};

template <>
struct MetaTypeClassDeterminant<String> {
  template <typename SQL_TYPE_INFO>
  static auto isTargetClass(SQL_TYPE_INFO const& s) {
    return s.is_string();
  }
};

template <>
struct MetaTypeClassDeterminant<Array> {
  template <typename SQL_TYPE_INFO>
  static auto isTargetClass(SQL_TYPE_INFO const& s) {
    return s.is_array();
  }
};

template <>
struct MetaSwitchTraits<SQLTypeInfo> {
  static auto getType(SQLTypeInfo const& s) { return s.get_type(); }
};

template <typename T, T VALUE>
struct MetaType {
  using ResolvedType = T;
  static T const resolved_value = VALUE;
};

template <MetaTypeClassifications T>
struct MetaTypeClass {
  static MetaTypeClassifications const sql_type_class = T;
};

template <typename T>
using MetaTypeOptional = boost::optional<T>;  // Switch after C++17

using UncapturedMetaTypeOptional = MetaTypeOptional<UncapturedMetaType>;
using UncapturedMetaTypeClassOptional = MetaTypeOptional<UncapturedMetaTypeClass>;

template <typename T, T VALUE>
using CapturedMetaTypeOptional = MetaTypeOptional<MetaType<T, VALUE>>;

template <MetaTypeClassifications T>
using CapturedMetaTypeClassOptional = MetaTypeOptional<MetaTypeClass<T>>;

template <typename T, T... VALUES_PACK>
class MetaTypeAny : public UncapturedMetaTypeOptional,
                    public CapturedMetaTypeOptional<T, VALUES_PACK>... {};

template <MetaTypeClassifications... CLASSIFICATIONS_PACK>
class MetaTypeClassAny : public UncapturedMetaTypeClassOptional,
                         public CapturedMetaTypeClassOptional<CLASSIFICATIONS_PACK>... {};

template <typename T, T... VALUES_PACK>
class MetaTypeFactory {
 public:
  using ResolvedType = T;
  using MetaTypeContainer = MetaTypeAny<ResolvedType, VALUES_PACK...>;

  template <typename SOURCE_TYPE>
  static auto getMetaType(SOURCE_TYPE&& source_type) {
    MetaTypeContainer return_value;
    resolveType<VALUES_PACK...>(return_value,
                                MetaSwitchTraits<SOURCE_TYPE>::getType(source_type));
    return return_value;
  }

 private:
  template <ResolvedType VALUE>
  static auto resolveType(MetaTypeContainer& return_value,
                          ResolvedType const switch_value) {
    if (switch_value == VALUE) {
      static_cast<CapturedMetaTypeOptional<ResolvedType, VALUE>&>(return_value) =
          MetaType<ResolvedType, VALUE>();
    }
    static_cast<UncapturedMetaTypeOptional&>(return_value) = UncapturedMetaType();
  }

  template <ResolvedType FIRST_VALUE,
            ResolvedType SECOND_VALUE,
            ResolvedType... REMAINING_VALUES>
  static auto resolveType(MetaTypeContainer& return_value,
                          ResolvedType const switch_value) {
    if (switch_value == FIRST_VALUE) {
      static_cast<CapturedMetaTypeOptional<ResolvedType, FIRST_VALUE>&>(return_value) =
          MetaType<ResolvedType, FIRST_VALUE>();
      return;
    }
    resolveType<SECOND_VALUE, REMAINING_VALUES...>(return_value, switch_value);
  }
};

template <MetaTypeClassifications... CLASSIFICATIONS_PACK>
class MetaTypeClassFactory {
 public:
  using MetaTypeClassContainer = MetaTypeClassAny<CLASSIFICATIONS_PACK...>;

  template <typename SQL_TYPE_INFO>
  static auto getMetaTypeClass(SQL_TYPE_INFO const& sql_type_info)
      -> MetaTypeClassContainer {
    MetaTypeClassContainer return_value;
    resolveClassification<SQL_TYPE_INFO, CLASSIFICATIONS_PACK...>(return_value,
                                                                  sql_type_info);
    return return_value;
  }

 private:
  template <typename SQL_TYPE_INFO, MetaTypeClassifications FIRST_TYPE>
  static void resolveClassification(MetaTypeClassContainer& return_value,
                                    SQL_TYPE_INFO const& sql_type_info) {
    if (MetaTypeClassDeterminant<FIRST_TYPE>::isTargetClass(sql_type_info)) {
      static_cast<CapturedMetaTypeClassOptional<FIRST_TYPE>&>(return_value) =
          MetaTypeClass<FIRST_TYPE>();
      return;
    }
    static_cast<UncapturedMetaTypeClassOptional&>(return_value) =
        UncapturedMetaTypeClass();
  }

  template <typename SQL_TYPE_INFO,
            MetaTypeClassifications FIRST_TYPE,
            MetaTypeClassifications SECOND_TYPE,
            MetaTypeClassifications... REMAINING_TYPES>
  static void resolveClassification(MetaTypeClassContainer& return_value,
                                    SQL_TYPE_INFO const& sql_type_info) {
    if (MetaTypeClassDeterminant<FIRST_TYPE>::isTargetClass(sql_type_info)) {
      static_cast<CapturedMetaTypeClassOptional<FIRST_TYPE>>(return_value) =
          MetaTypeClass<FIRST_TYPE>();
      return;
    }
    resolveClassification<SQL_TYPE_INFO, SECOND_TYPE, REMAINING_TYPES...>(return_value,
                                                                          sql_type_info);
  }
};

inline namespace Cpp14 {
struct Applicator {
  template <class FUNCTION_TYPE, class TUPLE_TYPE, std::size_t... I>
  decltype(auto) internalApplyImpl(FUNCTION_TYPE&& f,
                                   TUPLE_TYPE&& t,
                                   std::index_sequence<I...>) {
    return f(std::get<I>(std::forward<TUPLE_TYPE>(t))...);
  }

  template <class FUNCTION_TYPE, class TUPLE_TYPE>
  decltype(auto) internalApply(FUNCTION_TYPE&& f, TUPLE_TYPE&& t) {
    return internalApplyImpl(
        std::forward<FUNCTION_TYPE>(f),
        std::forward<TUPLE_TYPE>(t),
        std::make_index_sequence<
            std::tuple_size<std::remove_reference_t<TUPLE_TYPE>>::value>{});
  }
};

template <template <class> class SPECIALIZED_HANDLER,
          typename T,
          T... HANDLED_VALUES_PACK>
class MetaTypeHandler : protected Applicator {
 public:
  using ResolvedType = T;
  static constexpr std::size_t handled_type_count = sizeof...(HANDLED_VALUES_PACK);

  template <typename META_TYPE, typename... ARG_PACK>
  decltype(auto) operator()(META_TYPE const& meta_type, ARG_PACK&&... args) {
    using ArgumentPackaging = decltype(std::forward_as_tuple(args...));
    return handleMetaType<META_TYPE, ArgumentPackaging, HANDLED_VALUES_PACK...>(
        meta_type, std::forward_as_tuple(args...));
  }

 private:
  template <typename META_TYPE, typename ARG_PACKAGING, ResolvedType VALUE>
  decltype(auto) handleMetaType(META_TYPE const& meta_type,
                                ARG_PACKAGING&& arg_packaging) {
    using InspectionValue = MetaType<ResolvedType, VALUE>;
    using CastAssistType =
        CapturedMetaTypeOptional<ResolvedType, InspectionValue::resolved_value>;

    auto const& lhs_ref = static_cast<CastAssistType const&>(meta_type);
    if (!lhs_ref) {
      return internalApply(SPECIALIZED_HANDLER<UncapturedMetaType>(), arg_packaging);
    }
    return internalApply(SPECIALIZED_HANDLER<InspectionValue>(), arg_packaging);
  };

  template <typename META_TYPE,
            typename ARG_PACKAGING,
            ResolvedType FIRST_VALUE,
            ResolvedType SECOND_VALUE,
            ResolvedType... REMAINING_VALUES>
  decltype(auto) handleMetaType(META_TYPE const& meta_type,
                                ARG_PACKAGING&& arg_packaging) {
    using InspectionValue = MetaType<ResolvedType, FIRST_VALUE>;
    using CastAssistType =
        CapturedMetaTypeOptional<ResolvedType, InspectionValue::resolved_value>;

    auto const& lhs_ref = static_cast<CastAssistType const&>(meta_type);
    if (!lhs_ref) {
      return handleMetaType<META_TYPE, ARG_PACKAGING, SECOND_VALUE, REMAINING_VALUES...>(
          meta_type, std::forward<ARG_PACKAGING&&>(arg_packaging));
      return;
    }
    return internalApply(SPECIALIZED_HANDLER<InspectionValue>(), arg_packaging);
  };
};

template <template <class> class SPECIALIZED_HANDLER,
          MetaTypeClassifications... HANDLED_TYPE_CLASSES_PACK>
class MetaTypeClassHandler : protected Applicator {
 public:
  using TypeList = std::tuple<MetaTypeClass<HANDLED_TYPE_CLASSES_PACK>...>;
  static constexpr std::size_t handled_type_count = sizeof...(HANDLED_TYPE_CLASSES_PACK);

  template <typename META_TYPE_CLASS, typename... ARG_PACK>
  void operator()(META_TYPE_CLASS& meta_type_class, ARG_PACK&&... args) {
    using ArgumentPackaging = decltype(std::forward_as_tuple(args...));
    handleMetaTypeClass<META_TYPE_CLASS, ArgumentPackaging, HANDLED_TYPE_CLASSES_PACK...>(
        meta_type_class, std::forward_as_tuple(args...));
  }

 private:
  template <typename META_TYPE_CLASS,
            typename ARG_PACKAGING,
            MetaTypeClassifications HANDLED_TYPE>
  void handleMetaTypeClass(META_TYPE_CLASS& meta_type_class,
                           ARG_PACKAGING&& arg_packaging) {
    using InspectionClass = MetaTypeClass<HANDLED_TYPE>;
    using CastAssistClass =
        CapturedMetaTypeClassOptional<InspectionClass::sql_type_class>;

    auto& lhs_ref = static_cast<CastAssistClass&>(meta_type_class);
    if (!lhs_ref) {
      internalApply(SPECIALIZED_HANDLER<UncapturedMetaTypeClassOptional>(),
                    arg_packaging);
      return;
    }
    internalApply(SPECIALIZED_HANDLER<InspectionClass>(), arg_packaging);
  }

  template <typename META_TYPE_CLASS,
            typename ARG_PACKAGING,
            MetaTypeClassifications FIRST_HANDLED_TYPE,
            MetaTypeClassifications SECOND_HANDLED_TYPE,
            MetaTypeClassifications... REMAINING_TYPES>
  void handleMetaTypeClass(META_TYPE_CLASS& meta_type_class,
                           ARG_PACKAGING&& arg_packaging) {
    using InspectionClass = MetaTypeClass<FIRST_HANDLED_TYPE>;
    using CastAssistClass =
        CapturedMetaTypeClassOptional<InspectionClass::sql_type_class>;

    auto& lhs_ref = static_cast<CastAssistClass&>(meta_type_class);
    if (!lhs_ref) {
      handleMetaTypeClass<META_TYPE_CLASS,
                          ARG_PACKAGING,
                          SECOND_HANDLED_TYPE,
                          REMAINING_TYPES...>(arg_packaging);
      return;
    }
    internalApply(SPECIALIZED_HANDLER<InspectionClass>(), arg_packaging);
  }
};

}  // namespace Cpp14

using GeoMetaTypeFactory =
    MetaTypeFactory<SQLTypes, kPOINT, kLINESTRING, kPOLYGON, kMULTIPOLYGON>;
using FullMetaTypeFactory = MetaTypeFactory<SQLTypes,
                                            kNULLT,
                                            kBOOLEAN,
                                            kCHAR,
                                            kVARCHAR,
                                            kNUMERIC,
                                            kDECIMAL,
                                            kINT,
                                            kSMALLINT,
                                            kFLOAT,
                                            kDOUBLE,
                                            kTIME,
                                            kTIMESTAMP,
                                            kBIGINT,
                                            kTEXT,
                                            kDATE,
                                            kARRAY,
                                            kINTERVAL_DAY_TIME,
                                            kINTERVAL_YEAR_MONTH,
                                            kPOINT,
                                            kLINESTRING,
                                            kPOLYGON,
                                            kMULTIPOLYGON,
                                            kTINYINT,
                                            kGEOMETRY,
                                            kGEOGRAPHY>;

using GeoMetaTypeClassFactory = MetaTypeClassFactory<Geometry>;
using StringMetaTypeClassFactory = MetaTypeClassFactory<String>;
using FullMetaTypeClassFactory = MetaTypeClassFactory<Geometry, Array, String>;

template <template <class> class SPECIALIZED_HANDLER>
using GeoMetaTypeHandler = MetaTypeHandler<SPECIALIZED_HANDLER,
                                           SQLTypes,
                                           kPOINT,
                                           kLINESTRING,
                                           kPOLYGON,
                                           kMULTIPOLYGON>;

template <template <class> class SPECIALIZED_HANDLER>
using GeoVsNonGeoClassHandler = MetaTypeClassHandler<SPECIALIZED_HANDLER, Geometry>;

}  // namespace Experimental

#endif
#endif
