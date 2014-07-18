#include "Parse.h"
#include "RA/parser.h"
#include "RA/ast/RA_Program.h"

using namespace Parse_Namespace;
using namespace RA_Namespace;

std::pair<bool, RelAlgNode*> RAParse::parse(const std::string &input, std::string &errMsg) {
    Parser parser;
    RelAlgNode *parseRoot = 0;
    std::string lastParsed;
    int numErrors = parser.parse(input, parseRoot, lastParsed);
    if (numErrors > 0) {
        errMsg = "Error at: " + lastParsed;
        return std::pair<bool, RelAlgNode*>(false, NULL);
    }
    return std::pair<bool, RelAlgNode*>(true, parseRoot);
}
