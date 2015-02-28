#ifndef QUERYENGINE_QUERYTEMPLATEGENERATOR_H
#define QUERYENGINE_QUERYTEMPLATEGENERATOR_H

#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>

#include <string>

std::string unique_name(const char* base_name, const bool is_nested);
llvm::Function* query_template(llvm::Module*, const size_t aggr_col_count, const bool is_nested, const bool hoist_literals);
llvm::Function* query_group_by_template(llvm::Module*, const size_t aggr_col_count, const bool is_nested,
                                        const bool hoist_literals, const bool use_fast_path, const size_t groups_buffer_size);

inline bool should_use_shared_memory(const bool use_fast_path, const size_t groups_buffer_size) {
  // TODO(alex): re-enable once we're able to use shared memory effectively,
  //             as of now it performs worse on flights dataset; will re-visit
  const size_t shared_mem_threshold { 0 };
  return use_fast_path && (groups_buffer_size <= shared_mem_threshold);
}

#endif  // QUERYENGINE_QUERYTEMPLATEGENERATOR_H
