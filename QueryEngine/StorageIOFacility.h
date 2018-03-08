#ifndef STORAGEIOFACILITY_H
#define STORAGEIOFACILITY_H

#include "Fragmenter/InsertOrderFragmenter.h"
#include "Shared/ConfigResolve.h"
#include "Shared/UpdelRoll.h"
#include <boost/variant.hpp>

template <typename FRAGMENTER_TYPE = Fragmenter_Namespace::InsertOrderFragmenter>
class DefaultIOFacet {
 public:
  using FragmenterType = FRAGMENTER_TYPE;
  using DeleteVictimOffsetList = std::vector<uint64_t>;

  template <typename CATALOG_TYPE, typename TABLE_TYPE, typename FRAGMENT_INDEX, typename VICTIM_OFFSET_LIST>
  void deleteColumns(CATALOG_TYPE const& cat,
                     TABLE_TYPE const& table_descriptor,
                     FRAGMENT_INDEX frag_index,
                     VICTIM_OFFSET_LIST& victims) {
    const auto fragmenter = dynamic_cast<Fragmenter_Namespace::InsertOrderFragmenter*>(table_descriptor->fragmenter);
    CHECK(fragmenter);
    ColumnDescriptor const* deleted_column_desc = cat.getDeletedColumn(table_descriptor);
    if (deleted_column_desc != nullptr) {
      typename FragmenterType::ModifyTransactionTracker transaction_tracker;
      fragmenter->updateColumn(&cat,
                               table_descriptor,
                               deleted_column_desc,
                               frag_index,
                               victims,
                               ScalarTargetValue(int64_t(1L)),
                               Data_Namespace::MemoryLevel::CPU_LEVEL,
                               transaction_tracker);
      transaction_tracker.commitUpdate();
    } else {
      LOG(INFO) << "Delete metadata column unavailable; skipping delete operation.";
    }
  }
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

  StorageIOFacility(ExecutorType* executor, CatalogType const& catalog) : executor_(executor), catalog_(catalog) {}

  UpdateCallback yieldDeleteCallback(TableDescriptorType const* table_descriptor) {
    auto callback = [this, table_descriptor](FragmentUpdaterType const& update_log) -> void {
      auto fragment_index(update_log.getFragmentIndex());
      DeleteVictimOffsetList victim_offsets;

      for (size_t i = 0; i < update_log.getEntryCount(); ++i) {
        auto const row(update_log.getEntryAt(i));
        CHECK(!row.empty());
        auto terminal_column_iter = std::prev(row.end());
        const auto scalar_tv = boost::get<ScalarTargetValue>(&*terminal_column_iter);
        CHECK(scalar_tv);

        // TODO:  Flag ppan's PR to stay with int64 instead of uint64
        uint64_t fragment_offset = static_cast<uint64_t>(*(boost::get<int64_t>(scalar_tv)));
        victim_offsets.push_back(fragment_offset);
      }
      IOFacility().deleteColumns(catalog_, table_descriptor, fragment_index, victim_offsets);

    };
    return callback;
  }

 private:
  ExecutorType* executor_;
  CatalogType const& catalog_;
};

#endif
