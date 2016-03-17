#ifndef QUERYENGINE_RELALGTRANSLATOR_H
#define QUERYENGINE_RELALGTRANSLATOR_H

#include <memory>
#include <unordered_map>
#include <vector>

namespace Analyzer {

class Expr;

}  // namespace Analyzer

namespace Catalog_Namespace {

class Catalog;

}  // namespace Catalog_Namespace

class RelAlgNode;

class RexAgg;

class RexScalar;

// For scan inputs, in_metainfo is empty.
std::shared_ptr<Analyzer::Expr> translate_scalar_rex(
    const RexScalar* rex,
    const std::unordered_map<const RelAlgNode*, int>& input_to_nest_level,
    const Catalog_Namespace::Catalog& cat);

std::shared_ptr<Analyzer::Expr> translate_aggregate_rex(
    const RexAgg* rex,
    const std::vector<std::shared_ptr<Analyzer::Expr>>& scalar_sources);

#endif  // QUERYENGINE_RELALGTRANSLATOR_H
