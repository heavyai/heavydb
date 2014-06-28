#include <iostream>

#include "visitor/XMLTranslator.h"

int main() {
    
    XMLTranslator XMLTransVisitor;
    
    Table tbl("student");
    
    
    BaseTableDef btd("DROP", &tbl);
    
    Schema sch(&btd);
    
    SQL sql(&sch);
    SQLList sqlList(&sql);
    
    Program prog(&sqlList);
    prog.accept(XMLTransVisitor);
    
}