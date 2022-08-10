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

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/unique_ptr.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/variant.hpp>
#include <boost/serialization/vector.hpp>

#include "QueryEngine/RelAlgDag.h"
#include "QueryEngine/RelAlgDagSerializer/serialization/ExecutionResultSerializer.h"
#include "QueryEngine/RelAlgDagSerializer/serialization/QueryHintSerializer.h"
#include "QueryEngine/RelAlgDagSerializer/serialization/RexWindowBoundSerializer.h"
#include "QueryEngine/RelAlgDagSerializer/serialization/SQLTypeInfoSerializer.h"
#include "QueryEngine/RelAlgDagSerializer/serialization/SortFieldSerializer.h"
#include "QueryEngine/RelAlgDagSerializer/serialization/StdOptionalSerializer.h"
#include "QueryEngine/RelAlgDagSerializer/serialization/TargetMetaInfoSerializer.h"
#include "Shared/scope.h"

/**
 * Simple macros defining a list of derived classes for RelAlgNode and Rex/RexScalar base
 * classes respectively. Derived classes are intended to be leaf-level classes, meaning
 * they themselves should not be inherited. If new derived classes are added that require
 * serialization, then it should be as simple as adding that class name in one of these
 * respective lists here and providing a specialization in its appropriate
 * RelAlgDagSerializer::serialize() static method.
 */

// NOTE: RelTranslatedJoin is not in this list as it is a RelAlgNode only created
// during query execution and therefore not relevant here as RelAlgDag serialization
// should only be performed before query execution to avoid having to serialize any query
// state
#define REL_ALG_NODE_DERIVED_CLASSES                                           \
  RelScan, RelProject, RelAggregate, RelJoin, RelFilter, RelLeftDeepInnerJoin, \
      RelCompound, RelSort, RelModify, RelTableFunction, RelLogicalValues,     \
      RelLogicalUnion

#define REX_DERIVED_CLASSES                                                  \
  RexAbstractInput, RexLiteral, RexOperator, RexSubQuery, RexInput, RexCase, \
      RexFunctionOperator, RexWindowFunctionOperator, RexRef, RexAgg

namespace {

/**
 * Type-trait utility used for SFINAE overloads
 */
template <class T, class... Ts>
struct is_any_class
    : std::bool_constant<(std::is_same_v<T, typename std::remove_cv_t<Ts>> || ...)> {};

/**
 * Type utility for tagging all RelAlgNode-related classes, including RelAlgDag, in SFINAE
 * overloads
 */
template <class T>
using is_rel_alg_node_class =
    is_any_class<T, RelAlgNode, ModifyManipulationTarget, REL_ALG_NODE_DERIVED_CLASSES>;

template <class T>
inline constexpr bool is_rel_alg_node_class_v = is_rel_alg_node_class<T>::value;

/**
 * Type utility for tagging all Rex/RexScalar-derived classes in SFINAE overloads
 */
template <class T>
using is_rex_class = is_any_class<T, Rex, RexScalar, REX_DERIVED_CLASSES>;

template <class T>
inline constexpr bool is_rex_class_v = is_rex_class<T>::value;

/**
 * Type utility for tagging all RelAlgNode and Rex/Rex-scalar-related classes
 */
template <class T>
struct all_serializable_rel_alg_classes
    : std::bool_constant<is_rel_alg_node_class_v<T> || is_rex_class_v<T> ||
                         std::is_same_v<T, RelAlgDag>> {};

template <class T>
inline constexpr bool all_serializable_rel_alg_classes_v =
    all_serializable_rel_alg_classes<T>::value;

}  // namespace

/**
 * Primary struct for serializing the RelAlgNode/Rex/RexScalar nodes in a RelAlgDag
 * instance. All RelAlgDag-related classes/structs that don't have public getter/setter
 * interfaces for serialization-dependent members need to friend this struct for private
 * access and provide a serialization specialization below.
 */
struct RelAlgDagSerializer {
  // forward-declaring a deserialize context and thread-local storage for it in order
  // to access a Catalog_Namespace::Catalog instance that will be used to populate
  // RelAlgDag components that are dependent on catalog items, such as table/column
  // descriptors.
  class RelAlgDagDeserializeContext;
  static thread_local std::unique_ptr<RelAlgDagDeserializeContext>
      rel_alg_dag_deserialize_context;

  /**
   * Creates a ScopeGuard to be used deserialization-scope only. Creates a deserialization
   * context, initialized by a Catalog_Namespace::Catalog instance.
   * @param cat The current db catalog
   */
  static ScopeGuard createContextScopeGuard(const Catalog_Namespace::Catalog& cat);

  /**
   * Gets the current thread-local db catalog. This is only called during deserialization
   * by RelAlgDag-related classes that have a Catalog dependence.
   */
  static const Catalog_Namespace::Catalog& getCatalog();

  /**
   * Primary serialization method for Rex/RexScalar-related classes.
   * If you create a new class that inherits from the Rex/RexScalar base class that
   * requires serialization, it needs to be added here.
   *
   * Making use of "if constexpr" statements to avoid the boilerplate of creating a new
   * templatized method for each class type. This also potentially allows for doing
   * version checking in one spot, if we ever need to version serialization.
   */
  template <class Archive,
            class RexClass,
            typename std::enable_if_t<is_rex_class_v<RexClass>>* = nullptr>
  static void serialize(Archive& ar, RexClass& obj, const unsigned int version) {
    if constexpr (std::is_same_v<Rex, RexClass>) {
      (ar & obj.hash_);
    } else if constexpr (std::is_same_v<RexScalar, RexClass>) {
      (ar & boost::serialization::base_object<Rex>(obj));
    } else if constexpr (std::is_same_v<RexAbstractInput, RexClass>) {
      (ar & boost::serialization::base_object<RexScalar>(obj));
      (ar & obj.in_index_);
    } else if constexpr (std::is_same_v<RexLiteral, RexClass>) {
      (ar & boost::serialization::base_object<RexScalar>(obj));
      (ar & obj.literal_);
      (ar & obj.type_);
      (ar & obj.target_type_);
      (ar & obj.scale_);
      (ar & obj.precision_);
      (ar & obj.target_scale_);
      (ar & obj.target_precision_);
    } else if constexpr (std::is_same_v<RexOperator, RexClass>) {
      (ar & boost::serialization::base_object<RexScalar>(obj));
      (ar & obj.op_);
      (ar & obj.operands_);
      (ar & obj.type_);
    } else if constexpr (std::is_same_v<RexSubQuery, RexClass>) {
      (ar & boost::serialization::base_object<RexScalar>(obj));
      (ar & obj.type_);

      // Execution result should not be set before serialization. If it is means
      // RelAlgExecutor got its hands on it first before serialization. This is not
      // advised. Serialization should happen before any RelAlgExecutor processing.
      CHECK(obj.result_);
      CHECK(*obj.result_ == nullptr);

      // BUT we still need to serialize the RexSubQuery::result_. It is a shared_ptr of a
      // shared_ptr. The outer shared ptr should always be defined, pointing to the
      // interior shared_ptr that should be null. The way it is designed, this 2-tiered
      // shared ptr acts as a link between RexSubQuery instances that were deep copied
      // from a parent. A result should not exist, but the link should, so we need to
      // serialize result_ (or find a better linking mechanism)
      (ar & obj.result_);

      (ar & obj.ra_);
    } else if constexpr (std::is_same_v<RexInput, RexClass>) {
      (ar & boost::serialization::base_object<RexAbstractInput>(obj));
      (ar & obj.node_);
    } else if constexpr (std::is_same_v<RexCase, RexClass>) {
      (ar & boost::serialization::base_object<RexScalar>(obj));
      (ar & obj.expr_pair_list_);
      (ar & obj.else_expr_);
    } else if constexpr (std::is_same_v<RexFunctionOperator, RexClass>) {
      (ar & boost::serialization::base_object<RexOperator>(obj));
      (ar & obj.name_);
    } else if constexpr (std::is_same_v<RexWindowFunctionOperator, RexClass>) {
      (ar & boost::serialization::base_object<RexFunctionOperator>(obj));
      (ar & obj.kind_);
      (ar & obj.partition_keys_);
      (ar & obj.order_keys_);
      (ar & obj.collation_);
      (ar & obj.frame_start_bound_);
      (ar & obj.frame_end_bound_);
      (ar & obj.is_rows_);
    } else if constexpr (std::is_same_v<RexRef, RexClass>) {
      (ar & boost::serialization::base_object<RexScalar>(obj));
      (ar & obj.index_);
    } else if constexpr (std::is_same_v<RexAgg, RexClass>) {
      (ar & boost::serialization::base_object<Rex>(obj));
      (ar & obj.agg_);
      (ar & obj.distinct_);
      (ar & obj.type_);
      (ar & obj.operands_);
    } else {
      static_assert(!sizeof(RexClass), "Unhandled Rex class during serialization.");
    }
  }

  /**
   * Utility method that registers polymorphic-derived classes with the boost archive.
   * Registration is needed to ensure derived classes referenced via polymorphic pointer
   * get properly designated for serialization. See:
   * https://www.boost.org/doc/libs/1_74_0/libs/serialization/doc/serialization.html#registration
   *
   * NOTE: the class types that need to be serialized are passed as a list of template
   * arguments
   */
  template <class Archive, class... RelAlgNodeClasses>
  static void registerClassesWithArchive(Archive& ar) {
    (ar.template register_type<RelAlgNodeClasses>(), ...);
  }

  /**
   * Primary serialization method for RelAlgNode-related classes, including the root
   * RelAlgDag class. If you create a new class that inherits from the RelAlgNode base
   * class that requires serialization, it needs to be added here.
   *
   * NOTE: Making use of "if constexpr" statements to avoid the boilerplate of creating a
   * new templatized method for each class type.
   */
  template <class Archive,
            class RelAlgClass,
            typename std::enable_if_t<is_rel_alg_node_class_v<RelAlgClass>>* = nullptr>
  static void serialize(Archive& ar, RelAlgClass& obj, const unsigned int version) {
    if constexpr (std::is_same_v<RelAlgNode, RelAlgClass>) {
      (ar & obj.inputs_);
      (ar & obj.id_);
      (ar & obj.hash_);
      (ar & obj.is_nop_);

      // NOTE: not serializing the id_in_plan_tree_, context_data_, targets_metainfo_,
      // dag_node_id_, query_plan_dag_, & query_plan_dag_hash_ members. They are only
      // needed for RelAlgExecutor pathways and not needed at the time serialization
      // is needed.
    } else if constexpr (std::is_same_v<RelScan, RelAlgClass>) {
      (ar & boost::serialization::base_object<RelAlgNode>(obj));

      // NOTE: we're not serializing anything in regard to the member RelScan::td_. The
      // table descriptor is instead a construction-dependent argument and will be
      // serialized as part of the save/load contruction data. See
      // boost::serialization::save_construct_data override below.
      (ar & obj.field_names_);
      (ar & obj.hint_applied_);
      (ar & obj.hints_);
    } else if constexpr (std::is_same_v<ModifyManipulationTarget, RelAlgClass>) {
      (ar & obj.is_update_via_select_);
      (ar & obj.is_delete_via_select_);
      (ar & obj.varlen_update_required_);
      (ar & obj.target_columns_);
      (ar & obj.force_rowwise_output_);

      // NOTE: we're not serializing table_descriptor_. The table descriptor is
      // instead a constructor-dependent argument and will be saved/loaded as part of
      // custom contructor data. See: boost::serializer::load_construct_data below for
      // more details.
    } else if constexpr (std::is_same_v<RelProject, RelAlgClass>) {
      (ar & boost::serialization::base_object<RelAlgNode>(obj));
      (ar & boost::serialization::base_object<ModifyManipulationTarget>(obj));
      (ar & obj.scalar_exprs_);
      (ar & obj.fields_);
      (ar & obj.hint_applied_);
      (ar & obj.hints_);
      (ar & obj.has_pushed_down_window_expr_);
    } else if constexpr (std::is_same_v<RelAggregate, RelAlgClass>) {
      (ar & boost::serialization::base_object<RelAlgNode>(obj));
      (ar & obj.groupby_count_);
      (ar & obj.agg_exprs_);
      (ar & obj.fields_);
      (ar & obj.hint_applied_);
      (ar & obj.hints_);
    } else if constexpr (std::is_same_v<RelJoin, RelAlgClass>) {
      (ar & boost::serialization::base_object<RelAlgNode>(obj));
      (ar & obj.condition_);
      (ar & obj.join_type_);
      (ar & obj.hint_applied_);
      (ar & obj.hints_);
    } else if constexpr (std::is_same_v<RelFilter, RelAlgClass>) {
      (ar & boost::serialization::base_object<RelAlgNode>(obj));
      (ar & obj.filter_);
    } else if constexpr (std::is_same_v<RelLeftDeepInnerJoin, RelAlgClass>) {
      (ar & boost::serialization::base_object<RelAlgNode>(obj));
      (ar & obj.condition_);
      (ar & obj.outer_conditions_per_level_);
      (ar & obj.original_filter_);
      (ar & obj.original_joins_);
    } else if constexpr (std::is_same_v<RelCompound, RelAlgClass>) {
      (ar & boost::serialization::base_object<RelAlgNode>(obj));
      (ar & boost::serialization::base_object<ModifyManipulationTarget>(obj));

      (ar & obj.filter_expr_);
      (ar & obj.groupby_count_);
      (ar & obj.agg_exprs_);
      (ar & obj.fields_);
      (ar & obj.is_agg_);
      (ar & obj.scalar_sources_);
      (ar & obj.target_exprs_);
      (ar & obj.hint_applied_);
      (ar & obj.hints_);
    } else if constexpr (std::is_same_v<RelSort, RelAlgClass>) {
      (ar & boost::serialization::base_object<RelAlgNode>(obj));
      (ar & obj.collation_);
      (ar & obj.limit_);
      (ar & obj.offset_);
      (ar & obj.empty_result_);
      (ar & obj.limit_delivered_);
    } else if constexpr (std::is_same_v<RelModify, RelAlgClass>) {
      (ar & boost::serialization::base_object<RelAlgNode>(obj));
      // NOTE: not serializing anything in regard to RelModify::catalog_ or
      // table_descriptor_ members. They will be used as constructor-dependent arguments
      // instead and will be saved/loaded with custom constuctor data. See:
      // RelAlgSerializer for more.
      (ar & obj.flattened_);
      (ar & obj.operation_);
      (ar & obj.target_column_list_);
    } else if constexpr (std::is_same_v<RelTableFunction, RelAlgClass>) {
      (ar & boost::serialization::base_object<RelAlgNode>(obj));
      (ar & obj.function_name_);
      (ar & obj.fields_);
      (ar & obj.col_inputs_);
      (ar & obj.table_func_inputs_);
      (ar & obj.target_exprs_);
    } else if constexpr (std::is_same_v<RelLogicalValues, RelAlgClass>) {
      (ar & boost::serialization::base_object<RelAlgNode>(obj));
      (ar & obj.tuple_type_);
      (ar & obj.values_);
    } else if constexpr (std::is_same_v<RelLogicalUnion, RelAlgClass>) {
      (ar & boost::serialization::base_object<RelAlgNode>(obj));
      (ar & obj.is_all_);
    } else {
      static_assert(!sizeof(RelAlgClass),
                    "Unhandled RelAlgNode class during serialization");
    }
  }

  /**
   * Primary serialization method for the RelAlgDag clas.
   */
  template <class Archive>
  static void serialize(Archive& ar, RelAlgDag& rel_alg_dag, const unsigned int version) {
    // Need to register all RelAlgNode and RexRexScalar-derived classes for
    // serialization. This is to ensure derived classes referenced via polymorphic
    // pointer get properly designated for serialization.
    registerClassesWithArchive<Archive, REL_ALG_NODE_DERIVED_CLASSES>(ar);
    registerClassesWithArchive<Archive, REX_DERIVED_CLASSES>(ar);

    // NOTE: we are not archiving RelTranslatedJoin as it is a RelAlgNode only created
    // during query execution and therefore not relevant here as the serialization
    // archive for the RelAlgDag should only be saved/loaded before query execution to
    // avoid having to serialize any query state

    // now archive relevant RelAlgDag members
    (ar & rel_alg_dag.build_state_);
    (ar & rel_alg_dag.nodes_);
    (ar & rel_alg_dag.subqueries_);
    (ar & rel_alg_dag.query_hint_);
    (ar & rel_alg_dag.global_hints_);
  }
};

namespace boost {
namespace serialization {

/**
 * boost::serialization::serialize overload for all RelAlgDag-related classes that require
 * serialization.
 *
 * NOTE: for proper overload resolution of boost::serialization::serialize
 * while maintaining templatization on the RelAlgDag-related class type, this function is
 * specialized on the Archive type. If other archive types are to be used, they need to be
 * specialized here.
 */
template <
    class RelAlgType,
    typename std::enable_if_t<all_serializable_rel_alg_classes_v<RelAlgType>>* = nullptr>
void serialize(boost::archive::text_iarchive& ar,
               RelAlgType& obj,
               const unsigned int version) {
  RelAlgDagSerializer::serialize(ar, obj, version);
}

template <
    class RelAlgType,
    typename std::enable_if_t<all_serializable_rel_alg_classes_v<RelAlgType>>* = nullptr>
void serialize(boost::archive::text_oarchive& ar,
               RelAlgType& obj,
               const unsigned int version) {
  RelAlgDagSerializer::serialize(ar, obj, version);
}

/**
 * Needed for boost::variant with a boost::blank item.
 */
template <class Archive>
void serialize(Archive& ar, boost::blank& blank, const unsigned int version) {
  // no-op. does nothing with an empty class
}

/*******************************************************************************
 * The following serializes constructor arguments for TableDescriptor-dependent
 * classes, which are RelScan, RelProject, RelCompound, & RelModify.
 *******************************************************************************/

/**
 * SFINAE helper for TableDescriptor-dependent classes
 */
template <class T>
struct is_catalog_rel_alg_node
    : std::bool_constant<std::is_same_v<RelScan, typename std::remove_cv_t<T>> ||
                         std::is_same_v<RelProject, typename std::remove_cv_t<T>> ||
                         std::is_same_v<RelCompound, typename std::remove_cv_t<T>> ||
                         std::is_same_v<RelModify, typename std::remove_cv_t<T>>> {};

template <class T>
inline constexpr bool is_catalog_rel_alg_node_v = is_catalog_rel_alg_node<T>::value;

/**
 * Saves constructor data for TableDescriptor-dependent classes by saving out the table
 * descriptor name. The table name seems like the best choice for a
 * synchonization-independent descriptor. It was not immediately obvious whether the table
 * id is the same across all nodes in a distributed cluster, for instance.
 *
 * NOTE: for proper overload resolution of boost::serialization::save_construct_data while
 * maintaining templatization on RelAlgNode type, this function is specialized on the
 * Archive type, in this case boost::archive::text_oarchive. If other archive types are to
 * be used, then they would need to have specializations added here.
 */
template <class RelAlgNodeType,
          typename std::enable_if_t<is_catalog_rel_alg_node_v<RelAlgNodeType>>* = nullptr>
inline void save_construct_data(boost::archive::text_oarchive& ar,
                                const RelAlgNodeType* node,
                                const unsigned int version) {
  auto* td = node->getTableDescriptor();
  if (td) {
    CHECK(!td->tableName.empty());
    ar << td->tableName;
  } else {
    // we need to serialize an empty string as deserialization will expect to see a
    // string. The empty string will indicate a null table descriptor. There are many
    // circumstances in which a catalog-dependent RelAlgNode might have a null
    // TableDescriptor. Generally speaking, RelScan and RelModify nodes require a valid
    // table descriptor. RelCompound and RelProject do not.
    ar << std::string();
  }
}

/**
 * Construction templates for TableDescriptor dependent classes
 */
template <class RelAlgNodeType>
inline void construct_catalog_rel_alg_node(RelAlgNodeType* node,
                                           const Catalog_Namespace::Catalog& cat,
                                           const TableDescriptor* td) {
  ::new (node) RelAlgNodeType(td);
}

/**
 * RelModify construction specialization, which requires a catalog reference
 */
inline void construct_catalog_rel_alg_node(RelModify* node,
                                           const Catalog_Namespace::Catalog& cat,
                                           const TableDescriptor* td) {
  ::new (node) RelModify(cat, td);
}

/**
 * Loads constructor data and instantiates TableDescriptor-dependent classes by loading
 * the table name and accessing its table descriptor from a thread-local catalog
 * reference.
 *
 * NOTE: for proper overload resolution of boost::serialization::load_construct_data while
 * maintaining templatization on RelAlgNode type, this function is specialized
 * on the Archive type, in this case boost::archive::text_iarchive. This would break if
 * other archives are used.
 */
template <
    class RelAlgNodeType,
    typename std::enable_if_t<is_catalog_rel_alg_node<RelAlgNodeType>::value>* = nullptr>
inline void load_construct_data(boost::archive::text_iarchive& ar,
                                RelAlgNodeType* node,
                                const unsigned int version) {
  std::string table_name;
  ar >> table_name;
  auto& cat = RelAlgDagSerializer::getCatalog();
  const TableDescriptor* td{nullptr};
  if (!table_name.empty()) {
    td = cat.getMetadataForTable(table_name);
  }
  construct_catalog_rel_alg_node(node, cat, td);
}

}  // namespace serialization
}  // namespace boost
