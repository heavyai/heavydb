/*
 * Copyright 2017 MapD Technologies, Inc.
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

#include "../../Execute.h"
#include "../../RelAlgExecutionDescriptor.h"

std::string Executor::renderRows(RenderInfo* render_query_data) {
  CHECK(false);
  return "";
}

int64_t Executor::getRowidForPixel(const int64_t x,
                                   const int64_t y,
                                   const std::string& session_id,
                                   const int render_widget_id,
                                   const int pixelRadius) {
  CHECK(false);
  return 0;
}

std::shared_ptr<ResultSet> Executor::renderLines(const std::string& queryStr,
                                                 const ExecutionResult& results,
                                                 const Catalog_Namespace::SessionInfo& session,
                                                 const int render_widget_id,
                                                 const rapidjson::Value& data_desc,
                                                 RenderInfo* render_query_data) {
  CHECK(false);
  return nullptr;
}

std::shared_ptr<ResultSet> Executor::renderPolygons(const std::string& queryStr,
                                                    const ExecutionResult& results,
                                                    const Catalog_Namespace::SessionInfo& session,
                                                    const int render_widget_id,
                                                    const rapidjson::Value& data_desc,
                                                    RenderInfo* render_query_data,
                                                    const std::string* render_config_json,
                                                    const std::string& poly_table_name) {
  CHECK(false);
  return nullptr;
}

std::shared_ptr<ResultSet> Executor::renderNonInSituData(const std::string& queryStr,
                                                         const ExecutionResult& results,
                                                         const Catalog_Namespace::SessionInfo& session,
                                                         const int render_widget_id,
                                                         const rapidjson::Value& data_desc,
                                                         RenderInfo* render_query_data,
                                                         const std::string* render_config_json) {
  CHECK(false);
  return nullptr;
}

std::vector<int32_t> Executor::getStringIds(const std::string& col_name,
                                            const std::vector<std::string>& col_vals,
                                            const ::QueryRenderer::QueryDataLayout* query_data_layout,
                                            const ResultSet* results,
                                            const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                                            const bool warn) const {
  CHECK(false);
  return {};
}
