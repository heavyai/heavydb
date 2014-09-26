/**
 * @file    TranslatorTest.cpp
 * @author  Steven Stewart <steve@map-d.com>
 *
 * An interactive program for testing the translation of SQL to an RA query plan tree.
 */
#include <iostream>
#include <string>
#include "../Planner.h"

using namespace std;
using namespace Plan_Namespace;

int main() {
    Planner planner;
    string sql;
    pair<int, string> err;
    
    do {
        // obtain user input
        cout << "mapd> ";
        getline(cin,sql);
        if (sql == "q")
            break;
        else sql = sql + "\n";
        
        // get query plan
        err = planner.makePlan(sql);
        
        if (err.first > 0)
            cout << err.second << endl;

    } while (true);
}
