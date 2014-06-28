#include <iostream>

#include "visitor/XMLTranslator.h"

int main() {
    
    // declare visitor class
    XMLTranslator XMLTransVisitor;
    
    // build SQL statement
    Table tbl("student");
    BaseTableDef btd("DROP", &tbl);
    Schema sch(&btd);
    SQL sql(&sch);
    SQLList sqlList(&sql);
    Program prog(&sqlList);
    
    // Pass visitor to accept method of root AST node
    prog.accept(XMLTransVisitor);
    
}
