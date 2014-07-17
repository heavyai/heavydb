#include "SQL/parser.h"
#include "SQL/visitor/Visitor.h"
#include "RA/visitor/XMLTranslatorRA.h"
#include "QPTranslator.h"
#include <iostream>
#include <string>

using namespace std;
using namespace SQL_Namespace;

ASTNode *parse_root = 0;
RelAlgNode *root = 0;
int main(int argc, char ** argv) {
	Parser parser;
    string sql;
    do {
        cout << "Enter sql statement: ";
        getline(cin,sql);
        if (sql == "q")
            break;
        ASTNode *parseRoot = 0;
        parser.parse(sql, parseRoot);
      
        //XMLTranslator xml;
        QPTranslator xml;
        
        // make sure the root is NULL before we accept!
        xml.resetRoot();

        if (parseRoot != 0)
            parseRoot->accept(xml); 
        cout << "*********** Visiting new tree*********************" << endl;
        XMLTranslatorRA xml2;

        root = xml.getRoot();
        if (root == NULL) 
            cout << "You failed. xml.getRoot() is " << root << ". Time to brush up on your grammar." << endl;
        else
            root->accept(xml2); 
    }
    while (1==1);
    cout << "After parse" << endl;
    
}
