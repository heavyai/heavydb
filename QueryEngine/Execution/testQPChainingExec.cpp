/**
 * @file	testQPChainingExec.cpp
 * @author	Steven Stewart <steve@map-d.com>
 */
#include <iostream>
#include <string>
#include <vector>
#include "../../Shared/types.h"
#include "../Parse/RA/parser.h"
#include "QPChainingExec.h"
//#include "../../DataMgr/Metadata/Catalog.h"

using namespace std;
using Execution_Namespace::QPChainingExec;
using RA_Namespace::RelAlgNode;

int main(int argc, char ** argv) {

    // Create a parser for RA and... do stuff
    RAParser parser;
    string relAlg;
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
        QPChainingExec executioner;
        if (parseRoot != 0) {
            parseRoot->accept(executioner); 
            std::pair<bool, std::string> insertErr = executioner.isError();
            if (insertErr.first == true) {
                cout << "Error: " << insertErr.second << std::endl;
            }
        }
    }
    while(true);
    cout << "Good-bye." << endl;
}
