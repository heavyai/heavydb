#include <utility>
#include "Parse.h"
#include "SQL/parser.h"
//#include "SQL/ast/Program.h"

using namespace Parse_Namespace;
using namespace SQL_Namespace;

std::pair<bool, ASTNode*> SQLParse::parse(const std::string &input, std::string &errMsg) {
    Parser parser;
    ASTNode *parseRoot = 0;
    std::string lastParsed;
    int numErrors = parser.parse(input, parseRoot, lastParsed);
    if (numErrors > 0) {
        errMsg = "Error at: " + lastParsed;
        return std::pair<bool, ASTNode*>(false, NULL);
    }
    return std::pair<bool, ASTNode*>(true, parseRoot);
}
