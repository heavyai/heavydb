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

namespace Experimental {

struct UncapturedMetaType {};
struct UncapturedMetaTypeClass {};

enum MetaTypeClassifications { Geometry, Array };

template <MetaTypeClassifications T>
struct MetaTypeClassDeterminant {
  template <typename SQL_TYPE_INFO>
  static auto isTargetClass(SQL_TYPE_INFO const& s) -> bool {
    return false;
  }
};

template <>
struct MetaTypeClassDeterminant<Geometry> {
  template <typename SQL_TYPE_INFO>
  static auto isTargetClass(SQL_TYPE_INFO const& s) -> bool {
    return s.is_geometry();
  }
};

template <>
struct MetaTypeClassDeterminant<Array> {
  template <typename SQL_TYPE_INFO>
  static auto isTargetClass(SQL_TYPE_INFO const& s) -> bool {
    return s.is_array();
  }
};

template <SQLTypes T>
struct MetaType {
  static SQLTypes const sql_type = T;
};

template <MetaTypeClassifications T>
struct MetaTypeClass {
  static MetaTypeClassifications const sql_type_class = T;
};

template <typename T>
using MetaTypeOptional = boost::optional<T>;  // Switch after C++17

using UncapturedMetaTypeOptional = MetaTypeOptional<UncapturedMetaType>;
using UncapturedMetaTypeClassOptional = MetaTypeOptional<UncapturedMetaTypeClass>;

template <SQLTypes T>
using CapturedMetaTypeOptional = MetaTypeOptional<MetaType<T>>;

template <MetaTypeClassifications T>
using CapturedMetaTypeClassOptional = MetaTypeOptional<MetaTypeClass<T>>;

template <SQLTypes... TYPE_PACK>
class MetaTypeAny : public UncapturedMetaTypeOptional,
                    public CapturedMetaTypeOptional<TYPE_PACK>... {};

template <MetaTypeClassifications... CLASSIFICATIONS_PACK>
class MetaTypeClassAny : public UncapturedMetaTypeClassOptional,
                         public CapturedMetaTypeClassOptional<CLASSIFICATIONS_PACK>... {};

template <SQLTypes... TYPE_PACK>
class MetaTypeFactory {
 public:
  using MetaTypeContainer = MetaTypeAny<TYPE_PACK...>;

  template <typename SQL_TYPE_INFO>
  static auto getMetaType(SQL_TYPE_INFO const& sql_type_info) -> MetaTypeContainer {
    MetaTypeContainer return_value;
    resolveType<TYPE_PACK...>(return_value, sql_type_info.get_type());
    return return_value;
  }

 private:
  template <SQLTypes FIRST_TYPE>
  static auto resolveType(MetaTypeContainer& return_value, SQLTypes const sql_type)
      -> void {
    if (sql_type == FIRST_TYPE) {
      static_cast<CapturedMetaTypeOptional<FIRST_TYPE>>(return_value) =
          MetaType<FIRST_TYPE>();
      return;
    }
    static_cast<UncapturedMetaTypeOptional>(return_value) = UncapturedMetaType();
  }

  template <SQLTypes FIRST_TYPE, SQLTypes SECOND_TYPE, SQLTypes... REMAINING_TYPES>
  static auto resolveType(MetaTypeContainer& return_value, SQLTypes const sql_type)
      -> void {
    if (sql_type == FIRST_TYPE) {
      static_cast<CapturedMetaTypeOptional<FIRST_TYPE>>(return_value) =
          MetaType<FIRST_TYPE>();
      return;
    }
    resolveType<SECOND_TYPE, REMAINING_TYPES...>(return_value, sql_type);
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

template <template <class> class META_TYPE_CLASS_HANDLER,
          MetaTypeClassifications... HANDLED_TYPE_CLASSES_PACK>
class MetaTypeClassHandler {
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
  // Needed until C++17; then we can just use std::apply()
  // Use a back-channel to retrieve the return value for now
  template <class FUNCTION_TYPE, class TUPLE_TYPE, std::size_t... I>
  void internalApplyImpl(FUNCTION_TYPE&& f, TUPLE_TYPE&& t, std::index_sequence<I...>) {
    f(std::get<I>(std::forward<TUPLE_TYPE>(t))...);
  }

  // Needed until C++17; then we can just use std::apply()
  // Use a back-channel to retrieve the return value for now
  template <class FUNCTION_TYPE, class TUPLE_TYPE>
  void internalApply(FUNCTION_TYPE&& f, TUPLE_TYPE&& t) {
    internalApplyImpl(std::forward<FUNCTION_TYPE>(f),
                      std::forward<TUPLE_TYPE>(t),
                      std::make_index_sequence<
                          std::tuple_size<std::remove_reference_t<TUPLE_TYPE>>::value>{});
  }

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
      internalApply(META_TYPE_CLASS_HANDLER<UncapturedMetaTypeClassOptional>(),
                    arg_packaging);
      return;
    }
    internalApply(META_TYPE_CLASS_HANDLER<InspectionClass>(), arg_packaging);
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
    internalApply(META_TYPE_CLASS_HANDLER<InspectionClass>(), arg_packaging);
  }
};

using GeoMetaTypeFactory = MetaTypeFactory<kPOINT, kLINESTRING, kPOLYGON, kMULTIPOLYGON>;
using FullMetaTypeFactory = MetaTypeFactory<kNULLT,
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
using FullMetaTypeClassFactory = MetaTypeClassFactory<Geometry, Array>;

template <template <class> class META_TYPE_CLASS_HANDLER>
using GeoVsNonGeoClassHandler = MetaTypeClassHandler<META_TYPE_CLASS_HANDLER, Geometry>;

}  // namespace Experimental

#endif
#endif
