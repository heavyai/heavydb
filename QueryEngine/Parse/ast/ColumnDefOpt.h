#ifndef COLUMN_DEF_OPT_NODE_H
#define COLUMN_DEF_OPT_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

class ColumnDefOpt : public ASTNode {
    
public:
    Literal* lit;
   // SearchCondition srchCon;
    Table* tbl;
    ColumnCommalist* colComList;
    int rule_Flag;

    /* rule_Flags: 
    0 NOT NULL
    1 NOT NULL PRIMARY KEY
    2 NOT NULL UNIQUE
    3 DEFAULT [literal]
    4 DEFAULT NULL
    5 DEFAULT USER
    6 CHECK [search condition]
    7 REFERENCES [table]
    8 REFERENCES [table] [column_commalist]
    */

    /**< Constructor */
    explicit ColumnDefOpt(int rF) : rule_Flag(rF), lit(NULL), tbl(NULL) {}
    ColumnDefOpt(int rF, Literal *n) : rule_Flag(rF), lit(n), tbl(NULL) {}
   // ColumnDefOpt(int rule_Flag, SearchCondition *n2) : rule_Flag(rF), lit(NULL), srchCon(n2), tbl(NULL), colComList(NULL) {}
    ColumnDefOpt(int rF, Table *n3) : rule_Flag(rF), lit(NULL), tbl(n3) {}
    ColumnDefOpt(int rF, Table *n3, ColumnCommalist *n4) : rule_Flag(rF), lit(NULL), tbl(n3), colComList(n4) {}

    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // COLUMN_DEF_OPT_NODE_H
