#ifndef STORAGEIOFACILITY_H
#define STORAGEIOFACILITY_H

#include "Fragmenter/InsertOrderFragmenter.h"
#include "TargetMetaInfo.h"
#include "UpdateCacheInvalidators.h"

#include "Shared/ConfigResolve.h"
#include "Shared/UpdelRoll.h"
#include <boost/variant.hpp>

template <typename FRAGMENTER_TYPE = Fragmenter_Namespace::InsertOrderFragmenter>
class DefaultIOFacet {
 public:
  using FragmenterType = FRAGMENTER_TYPE;
  using DeleteVictimOffsetList = std::vector<uint64_t>;
  using UpdateTargetOffsetList = std::vector<uint64_t>;
  using UpdateTargetTypeList = std::vector<TargetMetaInfo>;
  using UpdateTargetColumnNamesList = std::vector<std::string>;
  using TransactionLog = typename FragmenterType::ModifyTransactionTracker;
  using TransactionLogPtr = std::unique_ptr<TransactionLog>;
  using ColumnValidationFunction = std::function<bool(std::string const&)>;

  template <typename CATALOG_TYPE,
            typename TABLE_ID_TYPE,
            typename COLUMN_NAME_TYPE,
            typename FRAGMENT_ID_TYPE,
            typename FRAGMENT_OFFSET_LIST_TYPE,
            typename UPDATE_VALUES_LIST_TYPE,
            typename COLUMN_TYPE_INFO>
  static void updateColumn(CATALOG_TYPE const& cat,
                           TABLE_ID_TYPE const&& table_id,
                           COLUMN_NAME_TYPE const& column_name,
                           FRAGMENT_ID_TYPE const frag_id,
                           FRAGMENT_OFFSET_LIST_TYPE const& frag_offsets,
                           UPDATE_VALUES_LIST_TYPE const& update_values,
                           COLUMN_TYPE_INFO const& col_type_info,
                           TransactionLog& transaction_tracker) {
    auto const* table_descriptor = cat.getMetadataForTable(table_id);
    auto* fragmenter = table_descriptor->fragmenter;
    CHECK(fragmenter);
    auto const* target_column = cat.getMetadataForColumn(table_id, column_name);

    fragmenter->updateColumn(&cat,
                             table_descriptor,
                             target_column,
                             frag_id,
                             frag_offsets,
                             update_values,
                             col_type_info,
                             Data_Namespace::MemoryLevel::CPU_LEVEL,
                             transaction_tracker);
  }

  template <typename CATALOG_TYPE,
            typename TABLE_ID_TYPE,
            typename FRAGMENT_ID_TYPE,
            typename VICTIM_OFFSET_LIST,
            typename COLUMN_TYPE_INFO>
  static void deleteColumns(CATALOG_TYPE const& cat,
                            TABLE_ID_TYPE const&& table_id,
                            FRAGMENT_ID_TYPE const frag_id,
                            VICTIM_OFFSET_LIST& victims,
                            COLUMN_TYPE_INFO const& col_type_info,
                            TransactionLog& transaction_tracker) {
    auto const* table_descriptor = cat.getMetadataForTable(table_id);
    auto* fragmenter = table_descriptor->fragmenter;
    CHECK(fragmenter);

    auto const* deleted_column_desc = cat.getDeletedColumn(table_descriptor);
    if (deleted_column_desc != nullptr) {
      fragmenter->updateColumn(&cat,
                               table_descriptor,
                               deleted_column_desc,
                               frag_id,
                               victims,
                               ScalarTargetValue(int64_t(1L)),
                               col_type_info,
                               Data_Namespace::MemoryLevel::CPU_LEVEL,
                               transaction_tracker);
    } else {
      LOG(INFO) << "Delete metadata column unavailable; skipping delete operation.";
    }
  }

  template <typename CATALOG_TYPE, typename TABLE_DESCRIPTOR_TYPE>
  static std::function<bool(std::string const&)> yieldColumnValidator(CATALOG_TYPE const& cat,
                                                                      TABLE_DESCRIPTOR_TYPE const* table_descriptor) {
    return [&cat, table_descriptor](std::string const& column_name) -> bool {
      auto const* target_column = cat.getMetadataForColumn(table_descriptor->tableId, column_name);

      // The default IO facet currently only rejects none-encoded string columns
      auto column_type(target_column->columnType);
      if (column_type.is_string() && column_type.get_compression() == kENCODING_NONE)
        return false;
      return true;
    };
  };
};

template <typename EXECUTOR_TRAITS,
          typename IO_FACET = DefaultIOFacet<>,
          typename FRAGMENT_UPDATER = UpdateLogForFragment>
class StorageIOFacility {
 public:
  using ExecutorType = typename EXECUTOR_TRAITS::ExecutorType;
  using CatalogType = typename EXECUTOR_TRAITS::CatalogType;
  using FragmentUpdaterType = FRAGMENT_UPDATER;
  using UpdateCallback = typename FragmentUpdaterType::Callback;
  using IOFacility = IO_FACET;
  using TableDescriptorType = typename EXECUTOR_TRAITS::TableDescriptorType;
  using DeleteVictimOffsetList = typename IOFacility::DeleteVictimOffsetList;
  using UpdateTargetOffsetList = typename IOFacility::UpdateTargetOffsetList;
  using UpdateTargetTypeList = typename IOFacility::UpdateTargetTypeList;
  using UpdateTargetColumnNamesList = typename IOFacility::UpdateTargetColumnNamesList;
  using UpdateTargetColumnNameType = typename UpdateTargetColumnNamesList::value_type;
  using ColumnValidationFunction = typename IOFacility::ColumnValidationFunction;

  class TransactionParameters {
   public:
    typename IOFacility::TransactionLog& getTransactionTracker() { return transaction_tracker_; }
    void finalizeTransaction() { transaction_tracker_.commitUpdate(); }

   private:
    typename IOFacility::TransactionLog transaction_tracker_;
  };

  struct DeleteTransactionParameters : public TransactionParameters {
   public:
    DeleteTransactionParameters() {}

   private:
    DeleteTransactionParameters(DeleteTransactionParameters const& other) = delete;
    DeleteTransactionParameters& operator=(DeleteTransactionParameters const& other) = delete;
  };

  class UpdateTransactionParameters : public TransactionParameters {
   public:
    UpdateTransactionParameters(TableDescriptorType const* table_desc,
                                UpdateTargetColumnNamesList const& update_column_names,
                                UpdateTargetTypeList const& target_types)
        : table_descriptor_(table_desc), update_column_names_(update_column_names), targets_meta_(target_types){};

    typename UpdateTargetColumnNamesList::size_type getUpdateColumnCount() const { return update_column_names_.size(); }
    TableDescriptorType const* getTableDescriptor() const { return table_descriptor_; }
    UpdateTargetTypeList const& getTargetsMetaInfo() const { return targets_meta_; }
    typename UpdateTargetTypeList::size_type getTargetsMetaInfoSize() const { return targets_meta_.size(); }
    UpdateTargetColumnNamesList const& getUpdateColumnNames() const { return update_column_names_; }

   private:
    UpdateTransactionParameters(UpdateTransactionParameters const& other) = delete;
    UpdateTransactionParameters& operator=(UpdateTransactionParameters const& other) = delete;

    TableDescriptorType const* table_descriptor_;
    UpdateTargetColumnNamesList update_column_names_;
    UpdateTargetTypeList const& targets_meta_;
  };

  StorageIOFacility(ExecutorType* executor, CatalogType const& catalog) : executor_(executor), catalog_(catalog) {}

  ColumnValidationFunction yieldColumnValidator(TableDescriptorType const* table_descriptor) {
    return IOFacility::yieldColumnValidator(catalog_, table_descriptor);
  }

  UpdateCallback yieldUpdateCallback(UpdateTransactionParameters& update_parameters) {
    using OffsetVector = std::vector<uint64_t>;
    using ScalarTargetValueVector = std::vector<ScalarTargetValue>;

    auto callback = [this, &update_parameters](FragmentUpdaterType const& update_log) -> void {
      for (decltype(update_parameters.getUpdateColumnCount()) column_index = 0;
           column_index < update_parameters.getUpdateColumnCount();
           column_index++) {
        OffsetVector column_offsets;
        ScalarTargetValueVector scalar_target_values;

        column_offsets.reserve(update_log.getEntryCount());
        scalar_target_values.reserve(update_log.getEntryCount());

        for (decltype(update_log.getEntryCount()) row_index = 0; row_index < update_log.getEntryCount(); row_index++) {
          auto const row(update_log.getEntryAt(row_index));

          CHECK(!row.empty());
          CHECK(row.size() == update_parameters.getUpdateColumnCount() + 1);

          auto terminal_column_iter = std::prev(row.end());
          const auto frag_offset_scalar_tv = boost::get<ScalarTargetValue>(&*terminal_column_iter);
          CHECK(frag_offset_scalar_tv);

          column_offsets.push_back(static_cast<uint64_t>(*(boost::get<int64_t>(frag_offset_scalar_tv))));
          scalar_target_values.push_back(boost::get<ScalarTargetValue>(row[column_index]));
        }

        IOFacility::updateColumn(catalog_,
                                 update_log.getPhysicalTableId(),
                                 update_parameters.getUpdateColumnNames()[column_index],
                                 update_log.getFragmentId(),
                                 column_offsets,
                                 scalar_target_values,
                                 update_log.getColumnType(column_index),
                                 update_parameters.getTransactionTracker());
      }
    };

    return callback;
  }

  UpdateCallback yieldDeleteCallback(DeleteTransactionParameters& delete_parameters) {
    auto callback = [this, &delete_parameters](FragmentUpdaterType const& update_log) -> void {
      DeleteVictimOffsetList victim_offsets;
      victim_offsets.reserve(update_log.getEntryCount());

      for (size_t i = 0; i < update_log.getEntryCount(); ++i) {
        auto const row(update_log.getEntryAt(i));
        CHECK(!row.empty());
        auto terminal_column_iter = std::prev(row.end());
        const auto scalar_tv = boost::get<ScalarTargetValue>(&*terminal_column_iter);
        CHECK(scalar_tv);

        uint64_t fragment_offset = static_cast<uint64_t>(*(boost::get<int64_t>(scalar_tv)));
        victim_offsets.push_back(fragment_offset);
      }
      IOFacility::deleteColumns(catalog_,
                                update_log.getPhysicalTableId(),
                                update_log.getFragmentId(),
                                victim_offsets,
                                update_log.getColumnType(0),
                                delete_parameters.getTransactionTracker());
    };
    return callback;
  }

 private:
  ExecutorType* executor_;
  CatalogType const& catalog_;
};

#endif
