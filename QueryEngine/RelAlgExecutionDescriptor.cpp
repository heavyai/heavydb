#ifdef HAVE_CALCITE
#include "RelAlgExecutionDescriptor.h"

std::vector<RaExecutionDesc> get_execution_descriptors(const RelAlgNode* ra_node) {
  CHECK(ra_node);
  if (dynamic_cast<const RelScan*>(ra_node) || dynamic_cast<const RelJoin*>(ra_node)) {
    throw std::runtime_error("Query not supported yet");
  }
  CHECK_GT(ra_node->inputCount(), size_t(0));
  CHECK_EQ(size_t(1), ra_node->inputCount());
  std::vector<ForLoop> for_loops;
  const auto in_node = ra_node->getInput(0);
  if (dynamic_cast<const RelJoin*>(in_node)) {
    const auto join_node = static_cast<const RelJoin*>(in_node);
    CHECK_EQ(size_t(2), join_node->inputCount());
    std::vector<RaExecutionDesc> next_descriptors;
    for (size_t i = 0; i < join_node->inputCount(); ++i) {
      const auto join_input = join_node->getInput(i);
      for_loops.emplace_back(join_input);
      auto join_next_descriptors = dynamic_cast<const RelScan*>(join_input) ? std::vector<RaExecutionDesc>{}
                                                                            : get_execution_descriptors(join_input);
      std::move(
          std::begin(join_next_descriptors), std::end(join_next_descriptors), std::back_inserter(next_descriptors));
    }
    next_descriptors.emplace_back(for_loops, ra_node);
    return next_descriptors;
  }
  for_loops.emplace_back(in_node);
  auto next_descriptors =
      dynamic_cast<const RelScan*>(in_node) ? std::vector<RaExecutionDesc>{} : get_execution_descriptors(in_node);
  if (dynamic_cast<const RelSort*>(ra_node)) {
    CHECK(!next_descriptors.empty());
    next_descriptors.pop_back();
  }
  next_descriptors.emplace_back(for_loops, ra_node);
  return next_descriptors;
}

#endif  // HAVE_CALCITE
