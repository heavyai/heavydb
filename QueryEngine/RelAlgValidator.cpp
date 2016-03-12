#include "RelAlgValidator.h"
#include "RelAlgVisitor.h"

namespace {

class RelAlgPassThroughVisitor : public RelAlgVisitor<bool> {
 protected:
  bool defaultResult() const override { return true; }
};

}  // namespace

bool is_valid_rel_alg(const RelAlgNode* rel_alg) {
  RelAlgPassThroughVisitor null_visitor;
  return null_visitor.visit(rel_alg);
}
