#ifndef QUERYENGINE_QUERYTEMPLATEGENERATOR_H
#define QUERYENGINE_QUERYTEMPLATEGENERATOR_H

#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>

#include <string>

std::string unique_name(const char* base_name, const uint32_t query_id);
llvm::Function* query_template(llvm::Module*, const size_t aggr_col_count, const uint32_t query_id);
llvm::Function* query_group_by_template(llvm::Module*, const size_t aggr_col_count, const uint32_t query_id);

#endif  // QUERYENGINE_QUERYTEMPLATEGENERATOR_H
