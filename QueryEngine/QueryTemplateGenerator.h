#ifndef QUERYENGINE_QUERYTEMPLATEGENERATOR_H
#define QUERYENGINE_QUERYTEMPLATEGENERATOR_H

#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>

llvm::Function* query_template(llvm::Module*, const size_t aggr_col_count);
llvm::Function* query_group_by_template(llvm::Module*, const size_t aggr_col_count);

#endif  // QUERYENGINE_QUERYTEMPLATEGENERATOR_H
