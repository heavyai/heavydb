/**
 * @file    Planner.h
 * @author  Steven Stewart <steve@map-d.com>
 *
 * The planner transforms an SQL statement into a plan. It's quite handy.
 *
 * Different types of queries produce different types of Plan objects, each of
 * which implement an AbstractPlan interface.
 *
 * Presently, the Planner acts is mainly a wrapper around the Translator.
 * Additional functionality may be added at some future time.
 */
#ifndef QueryEngine_Plan_Planner_h
#define QueryEngine_Plan_Planner_h

#include <string>
#include "../Parse/RA/ast/RelAlgNode.h"
#include "../../DataMgr/Metadata/Catalog.h"
#include "Translator.h"

using namespace RA_Namespace;

namespace Plan_Namespace {
    
    /**
     * @class Planner
     * @brief Presently, a wrapper around a query translator.
     *
     * This class provides two methods: makePlan and checkError. The former
     * takes an sql string (any kind of sql statement), and produces a Plan
     * object. A Plan object can be of many different types, including
     * a QueryPlan (for a select), an InsertPlan (for an insert), etc.
     */
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
        AbstractPlan* makePlan(std::string sql, QueryStmtType &stmtType);
        
        /**
         * @brief Returns whether there is an error and, if so, an error message.
         * @return A bool indicating whether an error occurred, and a string for the error message.
         */
        std::pair<bool, std::string> checkError();
        
    private:
        Translator &tr_;        /// a reference to a Translator object
    };
    
} // Plan_Namespace

#endif // QueryEngine_Plan_Planner_h