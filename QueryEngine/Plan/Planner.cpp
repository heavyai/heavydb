/**
 * @file    Planner.cpp
 * @author  Steven Stewart <steve@map-d.com>
 */
#include "Planner.h"
#include "Translator.h"
#include "../Parse/SQL/parser.h"
#include "../Parse/RA/visitor/XMLTranslator.h"

namespace Plan_Namespace {
    
    Planner::Planner(Catalog &c) : c_(c) {
        // NOP
    }
    
    Planner::~Planner() {
        // NOP
    }
 
    std::pair<int, std::string> Planner::makePlan(std::string sql) {
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
        Translator tr(c_);
        queryPlan_ = tr.translate(parseRoot);
        if (tr.isError()) {
            return pair<int, std::string>(1, tr.errorMsg());
        }

        // print for debugging
        if (tr.getType() == QUERY_STMT) {
            XMLTranslator ra2xml;
            queryPlan_->accept(ra2xml);
        }
        
        // return (should be successful here)
        assert(numErrors == 0);
        return pair<int, std::string>(0, "");
    }
    
}
