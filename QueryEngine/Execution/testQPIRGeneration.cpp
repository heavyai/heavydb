/**
 * @file	testQPChainingExec.cpp
 * @author	Steven Stewart <steve@map-d.com>
 */
#include "../../Shared/types.h"
#include "../Parse/RA/parser.h"
#include "QPIRPrepper.h"
#include "QPIRGenerator.h"

#include <iostream>
#include <string>
#include <vector>
#include <map>

//#include "../../DataMgr/Metadata/Catalog.h"

using namespace std;
using Execution_Namespace::QPIRPrepper;
using Execution_Namespace::QPIRGenerator;
using RA_Namespace::RelAlgNode;

int main(int argc, char ** argv) {

    // Create a parser for RA and... do stuff
    RAParser parser;
    string relAlg;
    map <string,llvm::Module *> codeMap;
    do {
        cout << "mapd> ";
        getline(cin,relAlg);
        if (relAlg == "q")
            break;
        else relAlg = relAlg + "\n";

        RelAlgNode *parseRoot = 0;
        string lastParsed;
        int numErrors = parser.parse(relAlg, parseRoot,lastParsed);
        if (numErrors > 0) {
            cout << "Error at: " << lastParsed << endl;
            continue;
        }
        if (numErrors > 0)
            cout << "# Errors: " << numErrors << endl;
        if (parseRoot == NULL) printf("parseRoot is NULL\n");
        
        // Walk it!
        llvm::Module * code;
        if (parseRoot != 0) {
            QPIRPrepper irPrepper;
            parseRoot->accept(irPrepper); 
            string signatureString;
            irPrepper.getSignatureString(signatureString);
            cout << "Signature String: " << signatureString << endl;
            auto codeIt = codeMap.find(signatureString);
            if (codeIt != codeMap.end()) { // found 
                code = codeIt -> second;
                cout << "CODE CACHED" << endl;
                code -> dump();
            }
            else {
                cout << "Code not found" << endl;
                QPIRGenerator irGenerator(irPrepper.attributeNodes_,irPrepper.constantNodes_, irPrepper.projectNodes_);
                parseRoot->accept(irGenerator); 
                std::pair<bool, std::string> insertErr = irGenerator.isError();
                if (insertErr.first == true) {
                    cout << "Error: " << insertErr.second << std::endl;
                }
                else {
                    codeMap[signatureString] = irGenerator.getModule();
                    cout << "CODE GENERATED" << endl;
                    codeMap[signatureString] -> dump();
                }
            }
        }
    }
    while(true);
    cout << "Good-bye." << endl;
}
