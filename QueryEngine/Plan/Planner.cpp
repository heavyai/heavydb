/**
 * @file    Planner.cpp
 * @author  Steven Stewart <steve@map-d.com>
 */
#include "Planner.h"
#include "Translator.h"
#include "Annotator.h"
#include "../Parse/SQL/parser.h"
#include "../Parse/RA/visitor/XMLTranslator.h"

namespace Plan_Namespace {
    
    Planner::Planner() {
        // NOP
    }
    
    Planner::~Planner() {
        // NOP
    }
 
    std::pair<int, std::string> Planner::makePlan(std::string sql, bool annotate, bool typeCheck) {
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
        Translator tr;
        queryPlan_ = tr.translate(parseRoot);

        // print for debugging
        XMLTranslator ra2xml;
        queryPlan_->accept(ra2xml);
        
        // annotate (obtain metadata from Catalog for named relation and attribute nodes)
        if (annotate) {
            Annotator a;
            std::pair<int, std::string> err = a.annotate(queryPlan_);
            numErrors = err.first;
            errorMsg = err.second;
            if (numErrors > 0) {
                errorMsg = "Catalog error at '" + lastParsed + "'";
                return pair<int, std::string>(numErrors, errorMsg);
            }
        }
        
        // type check
        if (typeCheck) {
            if (numErrors > 0) {
                errorMsg = "Type mismatch error at '" + lastParsed + "'";
                return pair<int, std::string>(numErrors, errorMsg);
            }
        }
        
        // return (should be successful here)
        assert(numErrors == 0);
        return pair<int, std::string>(0, "");
    }
    
}
