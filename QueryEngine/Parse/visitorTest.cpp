#include <iostream>

#include "visitor/SimplePrinterVisitor.h"

#include "ast/Table.h"
#include "ast/Program.h"

int main() {
    
    SimplePrinterVisitor printVisitor;
        
    Program prog("sql");
    prog.accept(printVisitor);
        
    Table tbl("Student", "Teacher");
    tbl.accept(printVisitor);
    
}