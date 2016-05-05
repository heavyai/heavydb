#include "RelAlgValidator.h"
#include "RelAlgVisitor.h"

namespace {

class RelAlgRejectAggregateVisitor : public RelAlgVisitor<bool> {
 protected:
  bool defaultResult() const override { return true; }

  bool visitAggregate(const RelAggregate* agg) const override { return agg->isNop(); }

  virtual bool aggregateResult(const bool& aggregate, const bool& next_result) const override {
    return aggregate && next_result;
  }
};

}  // namespace

bool is_valid_rel_alg(const RelAlgNode* rel_alg) {
  RelAlgRejectAggregateVisitor reject_aggregate;
  return reject_aggregate.visit(rel_alg);
}
