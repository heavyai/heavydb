/**
 * Copyright 2020 OmniSci, Inc
 */

#include "DistributedHandler.h"
#include "MapDServer.h"
#include "QueryEngine/PendingExecutionClosure.h"
#include "gen-cpp/CalciteServer.h"
#include "gen-cpp/serialized_result_set_types.h"

#include <boost/core/null_deleter.hpp>

#if defined(HAVE_DISTRIBUTED_5_0)
#include "Distributed/ee/Cluster.h"
#endif

static std::atomic_flag execute_spin_lock_ = ATOMIC_FLAG_INIT;

namespace {

#ifdef MAPD_EDITION_EE
TQueryResult aggregate_execution_times(const std::vector<TQueryResult>& all_results) {
  for (const auto& result : all_results) {
    CHECK(result.row_set.rows.empty() && result.row_set.columns.empty());
  }
  const auto max_execution_time_it =
      std::max_element(all_results.begin(),
                       all_results.end(),
                       [](const TQueryResult& lhs, const TQueryResult& rhs) {
                         return lhs.execution_time_ms < rhs.execution_time_ms;
                       });
  const auto max_total_time_it =
      std::max_element(all_results.begin(),
                       all_results.end(),
                       [](const TQueryResult& lhs, const TQueryResult& rhs) {
                         return lhs.total_time_ms < rhs.total_time_ms;
                       });
  TQueryResult agggregated_result;
  agggregated_result.execution_time_ms = max_execution_time_it->execution_time_ms;
  agggregated_result.total_time_ms = max_total_time_it->total_time_ms;
  return agggregated_result;
}

std::vector<TColumnRange> column_ranges_to_thrift(
    const AggregatedColRange& column_ranges) {
  std::vector<TColumnRange> thrift_column_ranges;
  const auto& column_ranges_map = column_ranges.asMap();
  for (const auto& kv : column_ranges_map) {
    TColumnRange thrift_column_range;
    thrift_column_range.col_id = kv.first.col_id;
    thrift_column_range.table_id = kv.first.table_id;
    const auto& expr_range = kv.second;
    switch (expr_range.getType()) {
      case ExpressionRangeType::Integer:
        thrift_column_range.type = TExpressionRangeType::INTEGER;
        thrift_column_range.int_min = expr_range.getIntMin();
        thrift_column_range.int_max = expr_range.getIntMax();
        thrift_column_range.bucket = expr_range.getBucket();
        thrift_column_range.has_nulls = expr_range.hasNulls();
        break;
      case ExpressionRangeType::Float:
      case ExpressionRangeType::Double:
        thrift_column_range.type = expr_range.getType() == ExpressionRangeType::Float
                                       ? TExpressionRangeType::FLOAT
                                       : TExpressionRangeType::DOUBLE;
        thrift_column_range.fp_min = expr_range.getFpMin();
        thrift_column_range.fp_max = expr_range.getFpMax();
        thrift_column_range.has_nulls = expr_range.hasNulls();
        break;
      case ExpressionRangeType::Invalid:
        thrift_column_range.type = TExpressionRangeType::INVALID;
        break;
      default:
        CHECK(false);
    }
    thrift_column_ranges.push_back(thrift_column_range);
  }
  return thrift_column_ranges;
}

std::vector<TDictionaryGeneration> string_dictionary_generations_to_thrift(
    const StringDictionaryGenerations& dictionary_generations) {
  std::vector<TDictionaryGeneration> thrift_dictionary_generations;
  for (const auto& kv : dictionary_generations.asMap()) {
    TDictionaryGeneration thrift_dictionary_generation;
    thrift_dictionary_generation.dict_id = kv.first;
    thrift_dictionary_generation.entry_count = kv.second;
    thrift_dictionary_generations.push_back(thrift_dictionary_generation);
  }
  return thrift_dictionary_generations;
}

std::vector<TTableGeneration> table_generations_to_thrift(
    const TableGenerations& table_generations) {
  std::vector<TTableGeneration> thrift_table_generations;
  for (const auto& kv : table_generations.asMap()) {
    TTableGeneration table_generation;
    table_generation.table_id = kv.first;
    table_generation.start_rowid = kv.second.start_rowid;
    table_generation.tuple_count = kv.second.tuple_count;
    thrift_table_generations.push_back(table_generation);
  }
  return thrift_table_generations;
}
#else
void throw_distributed_disabled() {
  throw std::runtime_error(
      "Distributed mode is only available in the Enterprise Edition.");
}
#endif  // MAPD_EDITION_EE

}  // namespace

void convert_to_distributed_insert_data(
    const Catalog_Namespace::SessionInfo& parent_session_info,
    Fragmenter_Namespace::InsertData& src_data,
    TInsertData& dst_data) {
  auto& cat = parent_session_info.getCatalog();
  dst_data.db_id = src_data.databaseId;
  dst_data.table_id = src_data.tableId;
  dst_data.num_rows = src_data.numRows;
  dst_data.column_ids = src_data.columnIds;

  int num_cols = dst_data.column_ids.size();

  for (int idx = 0; idx < num_cols; idx++) {
    auto columnId = dst_data.column_ids[idx];
    DataBlockPtr& src_block = src_data.data[idx];
    {
      TDataBlockPtr block;
      dst_data.data.push_back(block);
    }
    TDataBlockPtr& dst_block = dst_data.data[idx];
    const ColumnDescriptor* cd = cat.getMetadataForColumn(dst_data.table_id, columnId);
    if (cd->columnType.is_geometry() ||
        (cd->columnType.is_string() &&
         cd->columnType.get_compression() == kENCODING_NONE)) {
      dst_block.__isset.var_len_data = true;
      for (auto& str : *src_block.stringsPtr) {
        TVarLen varlen;
        varlen.is_null = false;
        varlen.payload = str;
        dst_block.var_len_data.push_back(varlen);
      }
    } else if (cd->columnType.is_array()) {
      dst_block.__isset.var_len_data = true;
      for (ArrayDatum& arrayDatum : *src_block.arraysPtr) {
        TVarLen varlen;
        if (arrayDatum.is_null) {
          varlen.is_null = true;
        } else {
          varlen.is_null = false;
          varlen.payload = std::string(reinterpret_cast<const char*>(arrayDatum.pointer),
                                       arrayDatum.length);
        }
        dst_block.var_len_data.push_back(varlen);
      }
    } else {
      auto size = cd->columnType.get_logical_size();

      if (cd->columnType.is_string() &&
          cd->columnType.get_compression() == kENCODING_DICT) {
        size = cd->columnType.get_size();
      }

      dst_block.__set_fixed_len_data(std::string(
          reinterpret_cast<const char*>(src_block.numbersPtr), dst_data.num_rows * size));
    }
  }
}

void MapDAggHandler::cluster_execute(TQueryResult& _return,
                                     QueryStateProxy query_state_proxy,
                                     const std::string& query_str,
                                     const bool column_format,
                                     const std::string& nonce,
                                     const int32_t first_n,
                                     const int32_t at_most_n,
                                     const SystemParameters& system_parameters) {
#ifdef MAPD_EDITION_EE

#if defined(HAVE_DISTRIBUTED_5_0)
  if (nullptr != Distributed::Cluster::instance() &&
      Distributed::Cluster::instance()->started()) {
    if (!Distributed::Cluster::instance()->checkAllLeaves()) {
      throw std::runtime_error("Failed to check all leaves ");
    }
  }
#endif

  auto const session_ptr = query_state_proxy.getQueryState().getConstSessionInfo();
  auto execute_calcite_permissable_query =
      [this, &query_state_proxy, &system_parameters](
          TQueryResult& _return,
          std::shared_ptr<Catalog_Namespace::SessionInfo const> const& session_ptr,
          const std::string& query_str,
          const bool acquire_locks,
          const ParserWrapper& pw,
          std::vector<std::vector<size_t>> outer_fragment_indices) {
        auto& leaf_aggregator = mapd_handler_->leaf_aggregator_;

        mapd_shared_lock<mapd_shared_mutex> execute_read_lock;
        if (acquire_locks) {
          execute_read_lock = mapd_shared_lock<mapd_shared_mutex>(
              *legacylockmgr::LockMgr<mapd_shared_mutex, bool>::getMutex(
                  legacylockmgr::ExecutorOuterLock, true));
        }

        TPlanResult parse_result;
        lockmgr::LockedTableDescriptors locks;
        std::tie(parse_result, locks) = mapd_handler_->parse_to_ra(
            query_state_proxy, query_str, {}, acquire_locks, system_parameters);
        const auto query_ra = parse_result.plan_result;

        ExecutionOptions eo = {false,
                               mapd_handler_->allow_multifrag_,
                               pw.isIRExplain(),
                               mapd_handler_->allow_loop_joins_,
                               g_enable_watchdog,
                               mapd_handler_->jit_debug_,
                               false,
                               g_enable_dynamic_watchdog,
                               g_dynamic_watchdog_time_limit,
                               false,
                               false,
                               mapd_handler_->system_parameters_.gpu_input_mem_limit,
                               g_enable_runtime_query_interrupt,
                               g_runtime_query_interrupt_frequency};
        const auto clock_begin = timer_start();
        auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID);
        const auto result = leaf_aggregator.execute(
            executor.get(), *session_ptr, query_ra, eo, nullptr, outer_fragment_indices);
        _return.total_time_ms = timer_stop(clock_begin);
        _return.execution_time_ms = _return.total_time_ms - result.rs->getQueueTime();
        return result;
      };

  ParserWrapper pw{query_str};
  auto& leaf_aggregator = mapd_handler_->leaf_aggregator_;
  if (pw.isCalcitePathPermissable()) {
    if (pw.isCalciteExplain()) {
      const auto query_ra =
          mapd_handler_
              ->parse_to_ra(query_state_proxy, query_str, {}, false, system_parameters)
              .first.plan_result;
      // return the ra as the result
      mapd_handler_->convert_explain(_return, ResultSet(query_ra), true);
      return;
    } else if (pw.isPlanExplain()) {
      const auto query_ra =
          mapd_handler_
              ->parse_to_ra(query_state_proxy, query_str, {}, false, system_parameters)
              .first.plan_result;
      auto& cat = session_ptr->getCatalog();
      auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID);
      RelAlgExecutor ra_executor(executor.get(), cat, query_ra);
      auto result = ra_executor.executeRelAlgQuery(CompilationOptions::defaults(),
                                                   ExecutionOptions::defaults(),
                                                   /*just_explain_plan=*/true,
                                                   nullptr);
      mapd_handler_->convert_explain(_return, *result.getDataPtr(), true);
      return;
    } else if (pw.isCalciteDdl()) {
      const auto query_ra =
          mapd_handler_
              ->parse_to_ra(query_state_proxy, query_str, {}, false, system_parameters)
              .first.plan_result;
      mapd_handler_->executeDdl(_return, query_ra, session_ptr);
      return;
    }

    std::unique_ptr<AggregatedResult> result{nullptr};
    {
      // TODO(adb): Use database locking to ensure atomicity across concurrent distributed
      // operations
      std::string query_session = "";
      auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID).get();
      if (g_enable_runtime_query_interrupt) {
        // a request of query execution without session id can happen, i.e., test query
        // if so, we turn back to the original way such that
        // we support runtime query interrupt without per-session management
        if (session_ptr != nullptr) {
          query_session = session_ptr->get_session_id();
          mapd_unique_lock<mapd_shared_mutex> session_write_lock(
              executor->getSessionLock());
          executor->addToQuerySessionList(query_session, session_write_lock);
          session_write_lock.unlock();
          // hybrid spinlock.  if it fails to acquire a lock, then
          // it sleeps {g_runtime_query_interrupt_frequency} millisecond.
          while (execute_spin_lock_.test_and_set(std::memory_order_acquire)) {
            // failed to get the spinlock: check whether query is interrupted
            mapd_shared_lock<mapd_shared_mutex> session_read_lock(
                executor->getSessionLock());
            bool isQueryInterrupted = executor->checkIsQuerySessionInterrupted(
                query_session, session_read_lock);
            session_read_lock.unlock();
            if (isQueryInterrupted) {
              mapd_unique_lock<mapd_shared_mutex> session_write_lock(
                  executor->getSessionLock());
              executor->removeFromQuerySessionList(query_session, session_write_lock);
              session_write_lock.unlock();
              VLOG(1) << "Kill the Interrupted pending query";
              throw std::runtime_error(
                  "Query execution has been interrupted (pending query)");
            }
            // here it fails to acquire the lock
            std::this_thread::sleep_for(
                std::chrono::milliseconds(g_runtime_query_interrupt_frequency));
          };
        }
        // currently, atomic_flag does not provide a way to get its current status,
        // i.e., spinlock.is_locked(), so we additionally lock the leafFlowMutex_
        // right after acquiring spinlock to let other part of the code can know
        // whether there exists a running query on the executor
      }
      mapd_unique_lock<mapd_shared_mutex> leafFlowLock(leafFlowMutex_);
      if (g_enable_runtime_query_interrupt) {
        // make sure to set the running session ID
        mapd_unique_lock<mapd_shared_mutex> session_write_lock(
            executor->getSessionLock());
        executor->invalidateQuerySession(session_write_lock);
        executor->setCurrentQuerySession(query_session, session_write_lock);
        session_write_lock.unlock();
      }

      ScopeGuard clearRuntimeInterruptStatus = [&executor] {
        if (g_enable_runtime_query_interrupt) {
          // reset the runtime query interrupt status
          mapd_shared_lock<mapd_shared_mutex> session_read_lock(
              executor->getSessionLock());
          std::string curSession = executor->getCurrentQuerySession(session_read_lock);
          session_read_lock.unlock();
          mapd_unique_lock<mapd_shared_mutex> session_write_lock(
              executor->getSessionLock());
          executor->removeFromQuerySessionList(curSession, session_write_lock);
          executor->invalidateQuerySession(session_write_lock);
          execute_spin_lock_.clear(std::memory_order_release);
          session_write_lock.unlock();
          executor->resetInterrupt();
          VLOG(1) << "RESET runtime query interrupt status of Executor " << executor;
        }
      };

      result = std::make_unique<AggregatedResult>(execute_calcite_permissable_query(
          _return, session_ptr, query_str, true, pw, {}));
    }
    CHECK(result);
    mapd_handler_->convert_rows(_return,
                                query_state_proxy,
                                result->targets_meta,
                                *(result->rs),
                                column_format,
                                first_n,
                                at_most_n);
  } else if (pw.is_update_dml && !pw.is_itas) {
    {
      mapd_unique_lock<mapd_shared_mutex> leafFlowLock(leafFlowMutex_);
      leaf_aggregator.leafCatalogConsistencyCheck(*session_ptr);
    }

    std::list<std::unique_ptr<Parser::Stmt>> parse_trees;
    DBHandler::parser_with_error_handler(query_str, parse_trees);

    auto stmt = parse_trees.front().get();
    Parser::InsertValuesStmt* insert_stmt = dynamic_cast<Parser::InsertValuesStmt*>(stmt);
    CHECK(insert_stmt);

    const auto td_with_lock =
        lockmgr::TableSchemaLockContainer<lockmgr::WriteLock>::acquireTableDescriptor(
            session_ptr->getCatalog(), *insert_stmt->get_table());
    const auto td = td_with_lock();
    CHECK(td);

    if (td->partitions == "REPLICATED") {
      // This is just a placeholder, we need real replication.
      auto all_results = leaf_aggregator.forwardQueryToLeaves(*session_ptr, query_str);
      _return = aggregate_execution_times(all_results);
    } else {
      auto leaf_idx = insert_stmt->determineLeafIndex(
          session_ptr->getCatalog(), mapd_handler_->leaf_aggregator_.leafCount());

      _return = leaf_aggregator.forwardQueryToLeaf(*session_ptr, query_str, leaf_idx);
    }
  } else {
    struct ThriftDistributedItasLeafConnector : public DistributedConnector {
     public:
      ThriftDistributedItasLeafConnector(
          DBHandler* handler,
          decltype(execute_calcite_permissable_query) execute_query)
          : mapd_handler_(handler), execute_calcite_permissable_query_(execute_query){};
      ~ThriftDistributedItasLeafConnector() override{};

      size_t getOuterFragmentCount(QueryStateProxy query_state_proxy,
                                   std::string& sql_query_string) override {
        auto const session_ptr = query_state_proxy.getQueryState().getConstSessionInfo();
        current_query_ = sql_query_string;
        outer_fragment_counts_ =
            mapd_handler_->leaf_aggregator_.query_get_outer_fragment_counts(
                *session_ptr, sql_query_string);

        size_t count = 0;
        for (auto leaf_count : outer_fragment_counts_) {
          count += leaf_count;
        }

        return count;
      }

      std::pair<size_t, size_t> getLeafFragmentIndex(size_t index) {
        size_t leaf_index = 0;
        size_t count = 0;
        for (; leaf_index < outer_fragment_counts_.size(); leaf_index++) {
          auto local_count = outer_fragment_counts_[leaf_index];

          if (index < local_count) {
            return {leaf_index, index};
          }

          index -= local_count;
          count += local_count;
        }

        return {0, count};
      }

      std::vector<AggregatedResult> query(
          QueryStateProxy query_state_proxy,
          std::string& sql_query_string,
          std::vector<size_t> outer_frag_indices) override {
        CHECK(sql_query_string == current_query_);
        auto const session_ptr = query_state_proxy.getQueryState().getConstSessionInfo();
        TQueryResult result;
        ParserWrapper pw{sql_query_string};

        std::vector<std::vector<size_t>> leaf_outer_fragment_indices;
        if (!outer_frag_indices.empty()) {
          for (auto count : outer_fragment_counts_) {
            leaf_outer_fragment_indices.push_back({count});
          }

          for (auto idx : outer_frag_indices) {
            auto leaf_idx = getLeafFragmentIndex(idx);
            if (leaf_idx.first < leaf_outer_fragment_indices.size()) {
              leaf_outer_fragment_indices[leaf_idx.first].push_back(leaf_idx.second);
            }
          }
        }

        return {execute_calcite_permissable_query_(result,
                                                   session_ptr,
                                                   sql_query_string,
                                                   false,
                                                   pw,
                                                   leaf_outer_fragment_indices)};
      }
      size_t leafCount() override { return mapd_handler_->leaf_aggregator_.leafCount(); }
      void insertDataToLeaf(const Catalog_Namespace::SessionInfo& parent_session_info,
                            const size_t leaf_idx,
                            Fragmenter_Namespace::InsertData& insert_data) override {
        TInsertData dst_data;
        convert_to_distributed_insert_data(parent_session_info, insert_data, dst_data);

        const auto td =
            parent_session_info.getCatalog().getMetadataForTable(insert_data.tableId);

        if (table_is_replicated(td)) {
          int leaf_count = mapd_handler_->leaf_aggregator_.leafCount();
          for (int leaf = 0; leaf < leaf_count; leaf++) {
            mapd_handler_->leaf_aggregator_.insertDataToLeaf(
                parent_session_info, leaf, dst_data);
          }
        } else {
          mapd_handler_->leaf_aggregator_.insertDataToLeaf(
              parent_session_info, leaf_idx, dst_data);
        }
      }

      void checkpoint(const Catalog_Namespace::SessionInfo& parent_session_info,
                      int tableId) override {
        auto& catalog = parent_session_info.getCatalog();
        auto dbId = catalog.getCurrentDB().dbId;
        mapd_handler_->leaf_aggregator_.checkpointLeaf(
            parent_session_info, dbId, tableId);
      }

      void rollback(const Catalog_Namespace::SessionInfo& parent_session_info,
                    int tableId) override {
        auto& catalog = parent_session_info.getCatalog();
        auto dbId = catalog.getCurrentDB().dbId;
        auto epoch = mapd_handler_->leaf_aggregator_.get_table_epochLeaf(
            parent_session_info, dbId, tableId);
        mapd_handler_->leaf_aggregator_.set_table_epochLeaf(
            parent_session_info, dbId, tableId, epoch);
      }

     private:
      DBHandler* mapd_handler_;
      decltype(execute_calcite_permissable_query) execute_calcite_permissable_query_;
      std::string current_query_;
      std::vector<size_t> outer_fragment_counts_;
    };

    if (pw.is_copy_to) {
      std::list<std::unique_ptr<Parser::Stmt>> parse_trees;
      auto shimmed_query_str = DBHandler::apply_copy_to_shim(query_str);
      DBHandler::parser_with_error_handler(shimmed_query_str, parse_trees);
      auto stmt = parse_trees.front().get();
      Parser::ExportQueryStmt* export_stmt = dynamic_cast<Parser::ExportQueryStmt*>(stmt);
      CHECK(export_stmt);

      // using thrift distributed leafs connector
      ThriftDistributedItasLeafConnector connector(mapd_handler_,
                                                   execute_calcite_permissable_query);
      export_stmt->leafs_connector_ = &connector;
      const auto clock_begin = timer_start();
      export_stmt->execute(*session_ptr);
      _return.total_time_ms += timer_stop(clock_begin);
      _return.execution_time_ms = _return.total_time_ms;
      return;
    }

    if (pw.is_copy || pw.is_validate) {
      // dont stop on validate or else we will not see the issues
      if (pw.is_copy) {
        mapd_unique_lock<mapd_shared_mutex> leafFlowLock(leafFlowMutex_);
        leaf_aggregator.leafCatalogConsistencyCheck(*session_ptr);
      }
      mapd_handler_->sql_execute_impl(_return,
                                      query_state_proxy,
                                      column_format,
                                      nonce,
                                      session_ptr->get_executor_device_type(),
                                      first_n,
                                      at_most_n);
      return;
    }

    {
      // lock is required to confirm the leaf and aggr executions all happen in same order
      mapd_unique_lock<mapd_shared_mutex> leafFlowLock(leafFlowMutex_);

      std::future<TQueryResult> aggregator_future{std::async(
          std::launch::async,
          [this,
           query_state_proxy,
           column_format,
           &nonce,
           &session_ptr,
           first_n,
           at_most_n] {
            TQueryResult result;
            mapd_handler_->sql_execute_impl(result,
                                            query_state_proxy,
                                            column_format,
                                            nonce,
                                            session_ptr->get_executor_device_type(),
                                            first_n,
                                            at_most_n);
            return result;
          })};
      if (LeafAggregator::queryShouldRunOnLeaves(query_str)) {
        auto all_results = leaf_aggregator.forwardQueryToLeaves(*session_ptr, query_str);
        all_results.push_back(aggregator_future.get());
        _return = aggregate_execution_times(all_results);
      } else {
        _return = aggregator_future.get();
      }
    }

    if (pw.is_ctas || pw.is_itas) {
      // the step above created the target table on all nodes
      // now the table needs data population
      std::list<std::unique_ptr<Parser::Stmt>> parse_trees;
      DBHandler::parser_with_error_handler(query_str, parse_trees);
      auto stmt = parse_trees.front().get();
      Parser::InsertIntoTableAsSelectStmt* itas =
          dynamic_cast<Parser::InsertIntoTableAsSelectStmt*>(stmt);
      CHECK(itas);

      try {
        // using thrift distributed leafs connector
        ThriftDistributedItasLeafConnector connector(mapd_handler_,
                                                     execute_calcite_permissable_query);
        itas->leafs_connector_ = &connector;
        const auto clock_begin = timer_start();
        itas->execute(*session_ptr);
        _return.total_time_ms += timer_stop(clock_begin);
        _return.execution_time_ms = _return.total_time_ms;
      } catch (...) {
        try {
          // drop the created table in case of an error...
          if (pw.is_ctas) {
            std::string drop_stmt = "DROP TABLE " + itas->get_table() + ";";
            TQueryResult drop_result;
            cluster_execute(drop_result,
                            query_state_proxy,
                            drop_stmt,
                            false,
                            "",
                            -1,
                            -1,
                            system_parameters);
          }
        } catch (...) {
          // eat it
        }

        throw;
      }
    }
  }
#else
  throw_distributed_disabled();
#endif  // MAPD_EDITION_EE
}

int64_t MapDLeafHandler::query_get_outer_fragment_count(const TSessionId& session,
                                                        const std::string& select_query) {
#ifdef MAPD_EDITION_EE
  const auto session_info = mapd_handler_->get_session_copy(session);
  const auto& catalog = session_info.getCatalog();

  auto session_copy = session_info;
  auto session_ptr = std::shared_ptr<Catalog_Namespace::SessionInfo>(
      &session_copy, boost::null_deleter());
  auto query_state = query_state::QueryState::create(session_ptr, select_query);

  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID);
  const auto device_type = ExecutorDeviceType::CPU;
  auto calcite_mgr = catalog.getCalciteMgr();

  // TODO MAT this should actually get the global or the session parameter for
  // view optimization
  const auto query_ra = calcite_mgr
                            ->process(query_state->createQueryStateProxy(),
                                      pg_shim(select_query),
                                      {},
                                      true,
                                      false,
                                      false,
                                      true)
                            .plan_result;
  CompilationOptions co = {
      device_type, true, ExecutorOptLevel::LoopStrengthReduction, false};
  ExecutionOptions eo = {
      false, true, false, true, false, false, false, false, 10000, false, false, 0.9};
  RelAlgExecutor ra_executor(executor.get(), catalog, query_ra);
  auto frag_count = ra_executor.getOuterFragmentCount(co, eo);

  return static_cast<int64_t>(frag_count);
#else
  throw_distributed_disabled();
#endif  // MAPD_EDITION_EE
};

void MapDLeafHandler::check_table_consistency(TTableMeta& _return,
                                              const TSessionId& session,
                                              const int32_t table_id) {
#ifdef MAPD_EDITION_EE
  const auto session_info = mapd_handler_->get_session_copy(session);
  const auto& cat = session_info.getCatalog();

  const auto td = cat.getMetadataForTable(table_id);
  if (!td) {
    throw std::runtime_error("Failed to find table with id " + std::to_string(table_id) +
                             " on leaf");
  }

  // check value of last table id in repo
  auto td_list = cat.getAllTableMetadata();
  if (td_list.empty()) {
    return;
  }
  td_list.sort(compare_td_id);

  _return.table_name = td->tableName;
  _return.num_cols = td->nColumns;
  _return.is_view = td->isView;
  _return.is_replicated = table_is_replicated(td);
  _return.shard_count = td->nShards;
  _return.max_rows = td->maxRows;
  _return.max_table_id = td_list.back()->tableId;
#else
  throw_distributed_disabled();
#endif  // MAPD_EDITION_EE
}

void MapDLeafHandler::start_query(TPendingQuery& _return,
                                  const TSessionId& leaf_session,
                                  const TSessionId& parent_session,
                                  const std::string& query_ra,
                                  const bool just_explain,
                                  const std::vector<int64_t>& outer_fragment_indices) {
#ifdef MAPD_EDITION_EE
  const auto session_info = mapd_handler_->get_session_copy(leaf_session);
  const auto& cat = session_info.getCatalog();

  CompilationOptions co =
      CompilationOptions::defaults(session_info.get_executor_device_type());
  co.with_dynamic_watchdog = g_enable_dynamic_watchdog;

  std::vector<size_t> outer_frags;
  for (auto idx : outer_fragment_indices) {
    outer_frags.push_back(static_cast<size_t>(idx));
  }

  ExecutionOptions eo = {false,
                         mapd_handler_->allow_multifrag_,
                         just_explain,
                         mapd_handler_->allow_loop_joins_,
                         g_enable_watchdog,
                         mapd_handler_->jit_debug_,
                         false,
                         g_enable_dynamic_watchdog,
                         g_dynamic_watchdog_time_limit,
                         false,
                         false,
                         mapd_handler_->system_parameters_.gpu_input_mem_limit,
                         g_enable_runtime_query_interrupt,
                         g_runtime_query_interrupt_frequency,
                         ExecutorType::Native,
                         outer_frags};
  RelAlgExecutionOptions ra_eo{co, eo, nullptr, 0};
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID,
                                        mapd_handler_->jit_debug_ ? "/tmp" : "",
                                        mapd_handler_->jit_debug_ ? "mapdquery" : "",
                                        mapd_handler_->system_parameters_);
  executor->setCatalog(&cat);
  mapd_unique_lock<mapd_shared_mutex> session_write_lock(executor->getSessionLock());
  executor->setCurrentQuerySession(parent_session, session_write_lock);
  session_write_lock.unlock();
  auto ra_executor = std::make_unique<RelAlgExecutor>(executor.get(), cat, query_ra);
  auto closure = PendingExecutionClosure::create(std::move(ra_executor), cat, ra_eo);
  _return.id = closure->getId();
  _return.column_ranges = column_ranges_to_thrift(closure->getColRangeCache());
  _return.dictionary_generations =
      string_dictionary_generations_to_thrift(closure->getStringDictionaryGenerations());
  _return.table_generations = table_generations_to_thrift(closure->getTableGenerations());
  _return.parent_session_id = parent_session;
#else
  throw_distributed_disabled();
#endif  // MAPD_EDITION_EE
}

#ifdef MAPD_EDITION_EE
TableGenerations table_generations_from_thrift(
    const std::vector<TTableGeneration>& thrift_table_generations) {
  TableGenerations table_generations;
  for (const auto& thrift_table_generation : thrift_table_generations) {
    table_generations.setGeneration(
        thrift_table_generation.table_id,
        TableGeneration{static_cast<size_t>(thrift_table_generation.tuple_count),
                        static_cast<size_t>(thrift_table_generation.start_rowid)});
  }
  return table_generations;
}
#endif  // MAPD_EDITION_EE

void MapDLeafHandler::execute_query_step(TStepResult& _return,
                                         const TPendingQuery& pending_query) {
#ifdef MAPD_EDITION_EE
  const auto first_step_result = PendingExecutionClosure::executeNextStep(
      pending_query.id,
      pending_query.parent_session_id,
      column_ranges_from_thrift(pending_query.column_ranges),
      string_dictionary_generations_from_thrift(pending_query.dictionary_generations),
      table_generations_from_thrift(pending_query.table_generations));
  const auto& result_set = first_step_result.result.getRows();
  const auto time_to_serialize =
      measure<>::execution([&]() { result_set->serialize(_return.serialized_rows); });

  LOG(INFO) << _return.serialized_rows.descriptor.entry_count
            << " rows being returned, total time to serialize was " << time_to_serialize
            << "ms, Time to compress was "
            << _return.serialized_rows.total_compression_time_ms << "ms."
            << " Uncompressed size was " << _return.serialized_rows.buffers_total_size
            << " bytes.";

  _return.execution_finished = first_step_result.is_outermost_query;
  _return.merge_type = first_step_result.merge_type == MergeType::Reduce
                           ? TMergeType::REDUCE
                           : TMergeType::UNION;
  _return.sharded = true;  // TODO(alex)
  _return.row_desc =
      mapd_handler_->convert_target_metainfo(first_step_result.result.getTargetsMeta());
  _return.node_id = first_step_result.node_id;
#else
  throw_distributed_disabled();
#endif  // MAPD_EDITION_EE
}

void MapDLeafHandler::broadcast_serialized_rows(const TSerializedRows& serialized_rows,
                                                const TRowDescriptor& row_desc,
                                                const TQueryId query_id) {
#ifdef MAPD_EDITION_EE

  std::unique_ptr<ResultSet> unserialized_result_set;

  unserialized_result_set = ResultSet::unserialize(
      serialized_rows, PendingExecutionClosure::getExecutor(query_id));

  std::shared_ptr<ResultSet> result_set(unserialized_result_set.release());

  const auto target_meta = target_meta_infos_from_thrift(row_desc);
  const auto subquery_result =
      std::make_shared<const ExecutionResult>(result_set, target_meta);
  PendingExecutionClosure::setCurrentSubqueryResult(query_id, subquery_result);
#else
  throw_distributed_disabled();
#endif  // MAPD_EDITION_EE
}

void MapDLeafHandler::flush_queue() {
#ifdef MAPD_EDITION_EE
  if (g_cluster && !mapd_handler_->leaf_aggregator_.leafCount()) {
    // check if there is a distributed query running already
    // this will remove it if it is once lock is available
    PendingExecutionClosure::flush();
  }
#else
  throw_distributed_disabled();
#endif  // MAPD_EDITION_EE
}
