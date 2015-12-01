#ifndef QUERYENGINE_CALCITEADAPTER_H
#define QUERYENGINE_CALCITEADAPTER_H

#include "../Catalog/Catalog.h"
#include "../Planner/Planner.h"

#include <string>

Planner::RootPlan* translate_query(const std::string& query, const Catalog_Namespace::Catalog& catalog);

#endif  // QUERYENGINE_CALCITEADAPTER_H
