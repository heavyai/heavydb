/**
 * @file    Planner.cpp
 * @author  Steven Stewart <steve@map-d.com>
 */
#include "Planner.h"
#include "Translator.h"
#include "../Parse/SQL/parser.h"
#include "../Parse/RA/visitor/XMLTranslator.h"

namespace Plan_Namespace {
 
    Planner::~Planner() {
        // NOP
    }
    
    std::pair<int, std::string> Planner::makePlan(std::string sql, RelAlgNode **plan, QueryStmtType &stmtType) {
        SQLParser parser;
        ASTNode *parseRoot = nullptr;
        string lastParsed;
        std::string errorMsg;
        int numErrors = 0;
        
        // parse SQL
        numErrors = parser.parse(sql, parseRoot, lastParsed);
        if (numErrors > 0) {
            errorMsg = "Syntax error at '" + lastParsed + "'";
            return pair<int, std::string>(numErrors, errorMsg);
        }
        
        // translate SQL AST to RA query plan
        *plan = tr_.translate(parseRoot);
        if (tr_.isError()) {
            return pair<int, std::string>(1, tr_.errorMsg());
        }
        
        // print for debugging
        if (tr_.getType() == QUERY_STMT) {
            XMLTranslator ra2xml;
            (*plan)->accept(ra2xml);
        }
        
        // return (should be successful here)
        assert(numErrors == 0);
        return std::pair<int, std::string>(0, "");
    }
} // Plan_Namespace
