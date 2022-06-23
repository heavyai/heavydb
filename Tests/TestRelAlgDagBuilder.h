/*
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

#include "QueryEngine/RelAlgDagBuilder.h"
#include "QueryEngine/RelLeftDeepInnerJoin.h"

class TestRelAlgDagBuilder : public RelAlgDag {
 public:
  struct AggDesc {
    AggDesc(SQLAgg agg_,
            bool distinct_,
            const SQLTypeInfo& type_,
            const std::vector<size_t>& operands_)
        : agg(agg_), distinct(distinct_), type(type_), operands(operands_) {}

    AggDesc(SQLAgg agg_,
            bool distinct_,
            SQLTypes type_,
            const std::vector<size_t>& operands_)
        : agg(agg_), distinct(distinct_), type(type_), operands(operands_) {}

    AggDesc(SQLAgg agg_, bool distinct_, SQLTypes type_, size_t operand)
        : agg(agg_), distinct(distinct_), type(type_), operands({operand}) {}

    AggDesc(SQLAgg agg_, SQLTypes type_, const std::vector<size_t>& operands_)
        : agg(agg_), distinct(false), type(type_), operands(operands_) {}

    AggDesc(SQLAgg agg_, SQLTypes type_, size_t operand)
        : agg(agg_), distinct(false), type(type_), operands({operand}) {}

    AggDesc(SQLAgg agg_) : agg(agg_), distinct(false), type(kINT) {
      CHECK(agg == kCOUNT);
    }

    SQLAgg agg;
    bool distinct;
    SQLTypeInfo type;
    std::vector<size_t> operands;
  };

  TestRelAlgDagBuilder(SchemaProviderPtr schema_provider, ConfigPtr config)
      : RelAlgDag(config), schema_provider_(schema_provider) {}
  ~TestRelAlgDagBuilder() override = default;

  RelAlgNodePtr addScan(const TableRef& table);
  RelAlgNodePtr addScan(int db_id, int table_id);
  RelAlgNodePtr addScan(int db_id, const std::string& table_name);

  RelAlgNodePtr addProject(RelAlgNodePtr input,
                           const std::vector<std::string>& fields,
                           const std::vector<int>& cols);
  RelAlgNodePtr addProject(RelAlgNodePtr input,
                           const std::vector<std::string>& fields,
                           std::vector<std::unique_ptr<const RexScalar>> cols);

  RelAlgNodePtr addProject(RelAlgNodePtr input, const std::vector<int>& cols);
  RelAlgNodePtr addProject(RelAlgNodePtr input,
                           std::vector<std::unique_ptr<const RexScalar>> cols);

  RelAlgNodePtr addAgg(RelAlgNodePtr input,
                       const std::vector<std::string>& fields,
                       size_t group_size,
                       std::vector<std::unique_ptr<const RexAgg>> aggs);
  RelAlgNodePtr addAgg(RelAlgNodePtr input,
                       const std::vector<std::string>& fields,
                       size_t group_size,
                       std::vector<AggDesc> aggs);

  RelAlgNodePtr addAgg(RelAlgNodePtr input,
                       size_t group_size,
                       std::vector<std::unique_ptr<const RexAgg>> aggs);
  RelAlgNodePtr addAgg(RelAlgNodePtr input, size_t group_size, std::vector<AggDesc> aggs);

  RelAlgNodePtr addSort(RelAlgNodePtr input,
                        const std::vector<SortField>& collation,
                        const size_t limit = 0,
                        const size_t offset = 0);

  RelAlgNodePtr addJoin(RelAlgNodePtr lhs,
                        RelAlgNodePtr rhs,
                        const JoinType join_type,
                        std::unique_ptr<const RexScalar> condition);

  RelAlgNodePtr addEquiJoin(RelAlgNodePtr lhs,
                            RelAlgNodePtr rhs,
                            const JoinType join_type,
                            size_t lhs_col_idx,
                            size_t rhs_col_idx);

  void setRoot(RelAlgNodePtr root);

  void finalize() {
    create_left_deep_join(nodes_);
    setRoot(nodes_.back());
  }

 private:
  RelAlgNodePtr addScan(TableInfoPtr table_info);

  std::vector<std::string> buildFieldNames(size_t count) const;

  SchemaProviderPtr schema_provider_;
};
