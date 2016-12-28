#include "QueryPhysicalInputsCollector.h"

#include "RelAlgAbstractInterpreter.h"
#include "RelAlgVisitor.h"
#include "RexVisitor.h"

namespace {

typedef std::unordered_set<PhysicalInput> PhysicalInputSet;

class RelAlgPhysicalInputsVisitor : public RelAlgVisitor<PhysicalInputSet> {
 public:
  PhysicalInputSet visitCompound(const RelCompound* compound) const override;
  PhysicalInputSet visitFilter(const RelFilter* filter) const override;
  PhysicalInputSet visitJoin(const RelJoin* join) const override;
  PhysicalInputSet visitProject(const RelProject* project) const override;

 protected:
  PhysicalInputSet aggregateResult(const PhysicalInputSet& aggregate,
                                   const PhysicalInputSet& next_result) const override;
};

class RexPhysicalInputsVisitor : public RexVisitor<PhysicalInputSet> {
 public:
  PhysicalInputSet visitInput(const RexInput* input) const override {
    const auto scan_ra = dynamic_cast<const RelScan*>(input->getSourceNode());
    if (!scan_ra) {
      return {};
    }
    const auto scan_td = scan_ra->getTableDescriptor();
    CHECK(scan_td);
    const int col_id = input->getIndex() + 1;
    const int table_id = scan_td->tableId;
    CHECK_GT(table_id, 0);
    return {{col_id, table_id}};
  }

  PhysicalInputSet visitSubQuery(const RexSubQuery* subquery) const override {
    const auto ra = subquery->getRelAlg();
    CHECK(ra);
    RelAlgPhysicalInputsVisitor visitor;
    return visitor.visit(ra);
  }

 protected:
  PhysicalInputSet aggregateResult(const PhysicalInputSet& aggregate,
                                   const PhysicalInputSet& next_result) const override {
    auto result = aggregate;
    result.insert(next_result.begin(), next_result.end());
    return result;
  }
};

PhysicalInputSet RelAlgPhysicalInputsVisitor::visitCompound(const RelCompound* compound) const {
  PhysicalInputSet result;
  for (size_t i = 0; i < compound->getScalarSourcesSize(); ++i) {
    const auto rex = compound->getScalarSource(i);
    CHECK(rex);
    RexPhysicalInputsVisitor visitor;
    const auto rex_phys_inputs = visitor.visit(rex);
    result.insert(rex_phys_inputs.begin(), rex_phys_inputs.end());
  }
  const auto filter = compound->getFilterExpr();
  if (filter) {
    RexPhysicalInputsVisitor visitor;
    const auto filter_phys_inputs = visitor.visit(filter);
    result.insert(filter_phys_inputs.begin(), filter_phys_inputs.end());
  }
  return result;
}

PhysicalInputSet RelAlgPhysicalInputsVisitor::visitFilter(const RelFilter* filter) const {
  const auto condition = filter->getCondition();
  CHECK(condition);
  RexPhysicalInputsVisitor visitor;
  return visitor.visit(condition);
}

PhysicalInputSet RelAlgPhysicalInputsVisitor::visitJoin(const RelJoin* join) const {
  const auto condition = join->getCondition();
  if (!condition) {
    return {};
  }
  RexPhysicalInputsVisitor visitor;
  return visitor.visit(condition);
}

PhysicalInputSet RelAlgPhysicalInputsVisitor::visitProject(const RelProject* project) const {
  PhysicalInputSet result;
  for (size_t i = 0; i < project->size(); ++i) {
    const auto rex = project->getProjectAt(i);
    CHECK(rex);
    RexPhysicalInputsVisitor visitor;
    const auto rex_phys_inputs = visitor.visit(rex);
    result.insert(rex_phys_inputs.begin(), rex_phys_inputs.end());
  }
  return result;
}

PhysicalInputSet RelAlgPhysicalInputsVisitor::aggregateResult(const PhysicalInputSet& aggregate,
                                                              const PhysicalInputSet& next_result) const {
  auto result = aggregate;
  result.insert(next_result.begin(), next_result.end());
  return result;
}

}  // namespace

std::unordered_set<PhysicalInput> get_physical_inputs(const RelAlgNode* ra) {
  RelAlgPhysicalInputsVisitor phys_inputs_visitor;
  return phys_inputs_visitor.visit(ra);
}
