#ifndef AST_SIMPLE_PRINTER_VISITOR_H
#define AST_SIMPLE_PRINTER_VISITOR_H

#include "Visitor.h"

#include "../ast/Program.h"
#include "../ast/Table.h"

#include <iostream>
using std::cout;

/**
 * @todo brief and detailed descriptions
 */
class SimplePrinterVisitor : public Visitor {

public:

    void visit(class Program *v) {
        cout << "<Program test='" << v->test << "'>\n";
    }

    void visit(class Table *v) {
        if (v->name2 != "")
            cout << "<Table name1='" << v->name1 << "' name2='" << v->name2 << "' />\n";
        else
            cout << "<Table name='" << v->name1 << "'/>\n";
    }

};

#endif // AST_SIMPLE_PRINTER_VISITOR_H