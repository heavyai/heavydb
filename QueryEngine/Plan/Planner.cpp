/**
 * @file    Planner.cpp
 * @author  Steven Stewart <steve@map-d.com>
 */
#include "Planner.h"
#include "Translator.h"
#include "../Parse/SQL/parser.h"

namespace Plan_Namespace {
 
    Planner::~Planner() {
        // NOP
    }
    
    AbstractPlan* Planner::makePlan(std::string sql, QueryStmtType &stmtType) {
        SQLParser parser;
        ASTNode *parseRoot = nullptr;
        string lastParsed;
        int numErrors = 0;
        
        // parse SQL
        numErrors = parser.parse(sql, parseRoot, lastParsed);
        if (numErrors > 0) {
            return nullptr;
        }
        
        // translate SQL AST to RA query plan
        AbstractPlan *queryPlan = tr_.translate(parseRoot);

        // get statement type
        stmtType = tr_.getType();
        
        if (tr_.isError())
            return nullptr;
        
        // return (should be successful here)
        assert(numErrors == 0);
        return queryPlan;
    }
    
    std::pair<bool, std::string> Planner::checkError() {
        return std::pair<bool, std::string>(tr_.isError(), tr_.errorMsg());
    }
    
} // Plan_Namespace
