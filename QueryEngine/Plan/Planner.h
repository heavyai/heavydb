/**
 * @file    Planner.h
 * @author  Steven Stewart <steve@map-d.com>
 *
 * The planner transforms an SQL statement into a query plan. Its responsibilities
 * include:
 *
 * 1) Verification: determine whether the SQL statement is meaningful
 * 2) Query Planning: construct a plan
 */
#ifndef QueryEngine_Plan_Planner_h
#define QueryEngine_Plan_Planner_h

#include <string>
#include "../Parse/RA/ast/RelAlgNode.h"
#include "../../DataMgr/Metadata/Catalog.h"
#include "Translator.h"

using namespace RA_Namespace;

namespace Plan_Namespace {
    
    class Planner {

    public:
        /// Constructor
        Planner(Translator &tr) : tr_(tr) {}
        
        /// Destructor
        ~Planner();
        
        /**
         * @brief Takes an sql string and makes a query plan.
         * @param sql The string containing the SQL statement.
         * @param plan Returns a reference to the root node of the resulting query plan
         * @param stmtType Returns the type of the generated plan
         * @return A pair (int - number of errors, string - error message).
         */
        std::pair<int, std::string> makePlan(std::string sql, RelAlgNode **plan, QueryStmtType &stmtType);
        
        
        int executeInsert(const RelAlgNode &plan);
        
    private:
        Translator &tr_; /// a reference to a Translator object
    };
    
} // Plan_Namespace

#endif // QueryEngine_Plan_Planner_h