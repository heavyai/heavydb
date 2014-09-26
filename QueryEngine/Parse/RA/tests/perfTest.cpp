/**
 * @file    ra2xml.cpp
 * @author  Steven Stewart <steve@map-d.com>
 */
#include <iostream>
#include <string>
#include "../parser.h"
#include "../visitor/Visitor.h"
#include "../visitor/XMLTranslator.h"
#include <boost/timer/timer.hpp>

using namespace std;
using namespace RA_Namespace;

int main(int argc, char ** argv) {
    RAParser parser;
    string input;
    
    do {
        cout << "mapd[RA]> ";
        getline(cin, input);
        
        // check for quit command
        if (input == "q")
            break;
        else
            input += "\n";
        
        // initialize variables
        RelAlgNode *parseTreeRoot = nullptr;
        string lastTokenParsed;
        int numErrors = 0;
        
        // set up timing
        const boost::timer::nanosecond_type oneSecond(1000000000LL);
        boost::timer::cpu_timer cpuTimer;
        int numQueries = 100000;
        
        // parse the input string
        for (int i = 0; i != numQueries && numErrors == 0; ++i) {
            numErrors = parser.parse(input, parseTreeRoot, lastTokenParsed);
        }
        double hostElapsedTime = double(cpuTimer.elapsed().user) / oneSecond * 1000000.0 /* microseconds */ / (double)numQueries /*numQueries*/;
        cout << "Query took: " << hostElapsedTime << " microseconds." <<  endl;
        
        // check for syntax error
        if (numErrors > 0) {
            cout << "Syntax error at token \"" + lastTokenParsed + "\"" << endl;
            continue;
        }
        
        // otherwise, print an XML representation of the parse tree
        //XMLTranslator xml;
        //parseTreeRoot->accept(xml);
        
    } while (true);
    
    cout << "Good-bye." << endl;
}
