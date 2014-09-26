/**
 * The main function in this file is used for testing a visitor of an
 * SQL AST.
 *
 *
 */
#include "../parser.h"
#include "../visitor/Visitor.h"
#include "../visitor/XMLTranslator.h"
#include <iostream>
#include <string>
#include <boost/timer/timer.hpp>

using namespace std;
using namespace SQL_Namespace;

int main(int argc, char ** argv) {
    SQLParser parser;
    string sql;
    do {
        cout << "mapd> ";
        getline(cin,sql);
        if (sql == "q")
            break;
        else sql = sql + "\n";
        
        ASTNode *parseRoot = 0;
        string lastParsed;
        int numErrors = 0;
        
        const boost::timer::nanosecond_type oneSecond(1000000000LL);
        boost::timer::cpu_timer cpuTimer;
        int numQueries = 100000;
        
        for (int i = 0; i != numQueries && numErrors == 0; ++i) {
            numErrors = parser.parse(sql, parseRoot,lastParsed);
        }
        double hostElapsedTime = double(cpuTimer.elapsed().user) / oneSecond * 1000000.0 /* microseconds */ / (double)numQueries /*numQueries*/;
        cout << "Query took: " << hostElapsedTime << " microseconds." <<  endl;
        
        if (numErrors > 0) {
            cout << "Error at: " << lastParsed << endl;
            continue;
        }
        if (numErrors > 0)
            cout << "# Errors: " << numErrors << endl;
    }
    while (true);
    cout << "Good-bye." << endl;
}
