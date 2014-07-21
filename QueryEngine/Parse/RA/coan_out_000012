#include "parser.h"
#include "visitor/Visitor.h"
#include "visitor/QPTranslator.h"
#include <iostream>
#include <string>

using namespace std;
using RA_Namespace::QPTranslator;

int main(int argc, char ** argv) {
    cout << "--------------------------------------------------\nMap-D, Relational Algebra to Query Plan Translator\n--------------------------------------------------" << endl;
    RAParser parser;
    string sql;
    do {
        cout << "mapd> ";
        getline(cin,sql);
        if (sql == "q")
            break;
        RelAlgNode *parseRoot = 0;
        string lastParsed;
        int numErrors = parser.parse(sql, parseRoot,lastParsed);
        if (numErrors > 0) {
            cout << "Error at: " << lastParsed << endl;
            continue;
        }
        
        //cout << "# Errors: " << numErrors << endl;
        QPTranslator qp;
        if (parseRoot != 0)
            parseRoot->accept(qp); 

    }
    while (1==1);
    cout << "Good-bye." << endl;
}
