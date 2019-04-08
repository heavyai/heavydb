#ifndef STORAGEIOFACILITY_H
#define STORAGEIOFACILITY_H

#include "Fragmenter/InsertOrderFragmenter.h"
#include "TargetMetaInfo.h"
#include "UpdateCacheInvalidators.h"

#include <boost/variant.hpp>
#include "Shared/ConfigResolve.h"
#include "Shared/ExperimentalTypeUtilities.h"
#include "Shared/UpdelRoll.h"
#include "Shared/likely.h"
#include "Shared/thread_count.h"

#include <future>

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
  static std::function<bool(std::string const&)> yieldColumnValidator(
      CATALOG_TYPE const& cat,
      TABLE_DESCRIPTOR_TYPE const* table_descriptor) {
    return [](std::string const& column_name) -> bool { return true; };
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

  using StringSelector = Experimental::MetaTypeClass<Experimental::String>;
  using NonStringSelector = Experimental::UncapturedMetaTypeClass;

  struct MethodSelector {
    static constexpr auto getEntryAt(StringSelector) {
      return &FragmentUpdaterType::getTranslatedEntryAt;
    }
    static constexpr auto getEntryAt(NonStringSelector) {
      return &FragmentUpdaterType::getEntryAt;
    }
  };

  class TransactionParameters {
   public:
    typename IOFacility::TransactionLog& getTransactionTracker() {
      return transaction_tracker_;
    }
    void finalizeTransaction() { transaction_tracker_.commitUpdate(); }

   private:
    typename IOFacility::TransactionLog transaction_tracker_;
  };

  struct DeleteTransactionParameters : public TransactionParameters {
   public:
    DeleteTransactionParameters() {}

   private:
    DeleteTransactionParameters(DeleteTransactionParameters const& other) = delete;
    DeleteTransactionParameters& operator=(DeleteTransactionParameters const& other) =
        delete;
  };

  class UpdateTransactionParameters : public TransactionParameters {
   public:
    UpdateTransactionParameters(TableDescriptorType const* table_desc,
                                UpdateTargetColumnNamesList const& update_column_names,
                                UpdateTargetTypeList const& target_types,
                                bool varlen_update_required)
        : table_descriptor_(table_desc)
        , update_column_names_(update_column_names)
        , targets_meta_(target_types)
        , varlen_update_required_(varlen_update_required){};

    auto getUpdateColumnCount() const { return update_column_names_.size(); }
    auto const* getTableDescriptor() const { return table_descriptor_; }
    auto const& getTargetsMetaInfo() const { return targets_meta_; }
    auto getTargetsMetaInfoSize() const { return targets_meta_.size(); }
    auto const& getUpdateColumnNames() const { return update_column_names_; }
    auto isVarlenUpdateRequired() const { return varlen_update_required_; }

   private:
    UpdateTransactionParameters(UpdateTransactionParameters const& other) = delete;
    UpdateTransactionParameters& operator=(UpdateTransactionParameters const& other) =
        delete;

    TableDescriptorType const* table_descriptor_;
    UpdateTargetColumnNamesList update_column_names_;
    UpdateTargetTypeList const& targets_meta_;
    bool varlen_update_required_ = false;
  };

  StorageIOFacility(ExecutorType* executor, CatalogType const& catalog)
      : executor_(executor), catalog_(catalog) {}

  ColumnValidationFunction yieldColumnValidator(
      TableDescriptorType const* table_descriptor) {
    return IOFacility::yieldColumnValidator(catalog_, table_descriptor);
  }

  UpdateCallback yieldUpdateCallback(UpdateTransactionParameters& update_parameters);
  UpdateCallback yieldDeleteCallback(DeleteTransactionParameters& delete_parameters);

 private:
  int normalized_cpu_threads() const { return cpu_threads() / 2; }

  ExecutorType* executor_;
  CatalogType const& catalog_;
};

template <typename EXECUTOR_TRAITS, typename IO_FACET, typename FRAGMENT_UPDATER>
typename StorageIOFacility<EXECUTOR_TRAITS, IO_FACET, FRAGMENT_UPDATER>::UpdateCallback
StorageIOFacility<EXECUTOR_TRAITS, IO_FACET, FRAGMENT_UPDATER>::yieldUpdateCallback(
    UpdateTransactionParameters& update_parameters) {
  using OffsetVector = std::vector<uint64_t>;
  using ScalarTargetValueVector = std::vector<ScalarTargetValue>;
  using RowProcessingFuturesVector = std::vector<std::future<uint64_t>>;

  if (update_parameters.isVarlenUpdateRequired()) {
    auto callback = [this,
                     &update_parameters](FragmentUpdaterType const& update_log) -> void {
      std::vector<const ColumnDescriptor*> columnDescriptors;

      for (size_t idx = 0; idx < update_parameters.getUpdateColumnNames().size(); idx++) {
        auto& column_name = update_parameters.getUpdateColumnNames()[idx];
        auto target_column =
            catalog_.getMetadataForColumn(update_log.getPhysicalTableId(), column_name);
        columnDescriptors.push_back(target_column);
      }

      auto td = catalog_.getMetadataForTable(update_log.getPhysicalTableId());
      auto* fragmenter = td->fragmenter;
      CHECK(fragmenter);

      fragmenter->updateColumns(
          &catalog_,
          td,
          update_log.getFragmentId(),
          columnDescriptors,
          update_log,
          update_parameters.getUpdateColumnCount(),  // last column of result set
          Data_Namespace::MemoryLevel::CPU_LEVEL,
          update_parameters.getTransactionTracker());
    };
    return callback;

  } else {
    auto callback = [this,
                     &update_parameters](FragmentUpdaterType const& update_log) -> void {
      auto rows_per_column = update_log.getEntryCount();
      if (rows_per_column == 0)
        return;

      OffsetVector column_offsets(rows_per_column);
      ScalarTargetValueVector scalar_target_values(rows_per_column);

      auto complete_row_block_size = rows_per_column / normalized_cpu_threads();
      auto partial_row_block_size = rows_per_column % normalized_cpu_threads();
      auto usable_threads = normalized_cpu_threads();
      if (UNLIKELY(rows_per_column < (unsigned)normalized_cpu_threads())) {
        complete_row_block_size = rows_per_column;
        partial_row_block_size = 0;
        usable_threads = 1;
      }

      auto process_rows =
          [&update_log, &update_parameters, &column_offsets, &scalar_target_values](
              auto type_tag,
              uint64_t column_index,
              uint64_t row_start,
              uint64_t row_count) -> uint64_t {
        uint64_t rows_processed = 0;
        for (uint64_t row_index = row_start; row_index < (row_start + row_count);
             row_index++, rows_processed++) {
          constexpr auto get_entry_method_sel(MethodSelector::getEntryAt(type_tag));
          auto const row((update_log.*get_entry_method_sel)(row_index));

          CHECK(!row.empty());
          CHECK(row.size() == update_parameters.getUpdateColumnCount() + 1);

          auto terminal_column_iter = std::prev(row.end());
          const auto frag_offset_scalar_tv =
              boost::get<ScalarTargetValue>(&*terminal_column_iter);
          CHECK(frag_offset_scalar_tv);

          column_offsets[row_index] =
              static_cast<uint64_t>(*(boost::get<int64_t>(frag_offset_scalar_tv)));
          scalar_target_values[row_index] =
              boost::get<ScalarTargetValue>(row[column_index]);
        }
        return rows_processed;
      };

      auto get_row_index = [complete_row_block_size](uint64_t thread_index) -> uint64_t {
        return (thread_index * complete_row_block_size);
      };

      // Iterate over each column
      for (decltype(update_parameters.getUpdateColumnCount()) column_index = 0;
           column_index < update_parameters.getUpdateColumnCount();
           column_index++) {
        RowProcessingFuturesVector row_processing_futures;
        row_processing_futures.reserve(usable_threads);

        auto thread_launcher = [&](auto const& type_tag) {
          for (unsigned i = 0; i < static_cast<unsigned>(usable_threads); i++)
            row_processing_futures.emplace_back(
                std::async(std::launch::async,
                           std::forward<decltype(process_rows)>(process_rows),
                           type_tag,
                           column_index,
                           get_row_index(i),
                           complete_row_block_size));
          if (partial_row_block_size) {
            row_processing_futures.emplace_back(
                std::async(std::launch::async,
                           std::forward<decltype(process_rows)>(process_rows),
                           type_tag,
                           column_index,
                           get_row_index(usable_threads),
                           partial_row_block_size));
          }
        };

        if (!update_log.getColumnType(column_index).is_string()) {
          thread_launcher(NonStringSelector());
        } else {
          thread_launcher(StringSelector());
        }

        uint64_t rows_processed(0);
        for (auto& t : row_processing_futures) {
          t.wait();
          rows_processed += t.get();
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
}

template <typename EXECUTOR_TRAITS, typename IO_FACET, typename FRAGMENT_UPDATER>
typename StorageIOFacility<EXECUTOR_TRAITS, IO_FACET, FRAGMENT_UPDATER>::UpdateCallback
StorageIOFacility<EXECUTOR_TRAITS, IO_FACET, FRAGMENT_UPDATER>::yieldDeleteCallback(
    DeleteTransactionParameters& delete_parameters) {
  using RowProcessingFuturesVector = std::vector<std::future<uint64_t>>;

  auto callback = [this,
                   &delete_parameters](FragmentUpdaterType const& update_log) -> void {
    auto rows_per_column = update_log.getEntryCount();
    if (rows_per_column == 0)
      return;
    DeleteVictimOffsetList victim_offsets(rows_per_column);

    auto complete_row_block_size = rows_per_column / normalized_cpu_threads();
    auto partial_row_block_size = rows_per_column % normalized_cpu_threads();
    auto usable_threads = normalized_cpu_threads();

    if (UNLIKELY(rows_per_column < (unsigned)normalized_cpu_threads())) {
      complete_row_block_size = rows_per_column;
      partial_row_block_size = 0;
      usable_threads = 1;
    }

    auto process_rows = [&update_log, &victim_offsets](uint64_t row_start,
                                                       uint64_t row_count) -> uint64_t {
      uint64_t rows_processed = 0;

      for (uint64_t row_index = row_start; row_index < (row_start + row_count);
           row_index++, rows_processed++) {
        auto const row(update_log.getEntryAt(row_index));
        __builtin_prefetch(row.data(), 0, 0);

        CHECK(!row.empty());
        auto terminal_column_iter = std::prev(row.end());
        const auto scalar_tv = boost::get<ScalarTargetValue>(&*terminal_column_iter);
        CHECK(scalar_tv);

        uint64_t fragment_offset =
            static_cast<uint64_t>(*(boost::get<int64_t>(scalar_tv)));
        victim_offsets[row_index] = fragment_offset;
      }
      return rows_processed;
    };

    auto get_row_index = [complete_row_block_size](uint64_t thread_index) -> uint64_t {
      return thread_index * complete_row_block_size;
    };

    RowProcessingFuturesVector row_processing_futures;
    row_processing_futures.reserve(usable_threads);

    for (unsigned i = 0; i < (unsigned)usable_threads; i++)
      row_processing_futures.emplace_back(
          std::async(std::launch::async,
                     std::forward<decltype(process_rows)>(process_rows),
                     get_row_index(i),
                     complete_row_block_size));
    if (partial_row_block_size)
      row_processing_futures.emplace_back(
          std::async(std::launch::async,
                     std::forward<decltype(process_rows)>(process_rows),
                     get_row_index(usable_threads),
                     partial_row_block_size));

    uint64_t rows_processed(0);
    for (auto& t : row_processing_futures) {
      t.wait();
      rows_processed += t.get();
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

#endif
