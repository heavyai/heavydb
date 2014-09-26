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

using namespace RA_Namespace;

namespace Plan_Namespace {
    
    class Planner {

    public:
        /// Constructor
        Planner();
        
        /// Destructor
        ~Planner();
        
        /**
         * @brief Takes an sql string and makes a query plan.
         * @param sql The string containing the SQL statement.
         * @param bool If true, the query plan is annotated with metadata from the catalog.
         * @param bool If true, the query plan is type checked.
         * @return A pair (int - number of erros, string - error message).
         */
        std::pair<int, std::string> makePlan(std::string sql, bool annotate = true, bool typeCheck = true);
        
        /**
         * @brief Returns a pointer to the root node of the RA query plan tree.
         */
        inline RelAlgNode* getPlan() const { return queryPlan_; }
        
    private:
        RelAlgNode *queryPlan_ = nullptr;
        
        /**
         * @brief Annotates the query plan with metadata obtained from the catalog
         * @return A pair (int - number of erros, string - error message).
         */
        std::pair<int, std::string> annotatePlan();
        
        /**
         * @brief Type checks a query plan.
         * @return A pair (int - number of erros, string - error message).
         */
        std::pair<int, std::string> typeCheck();
        
        
    };
    
} // Plan_Namespace

#endif // QueryEngine_Plan_Planner_h