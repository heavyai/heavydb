#ifndef TABLE_CONSTRAINT_DEF_NODE_H
#define TABLE_CONSTRAINT_DEF_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

class TableConstraintDef : public ASTNode {
    
public:

	int rule_Flag;
	/* rules are:
	0 UNIQUE
	1 PRIMARY KEY
	2 FOREIGN KEY _ REFERENCES
	3 CHECK */

    ColumnCommalist *colCom1;
    ColumnCommalist *colCom2;
    Table *tbl;

    /* constructor */
    explicit TableConstraintDef(int rF, ColumnCommalist *n) : rule_Flag(rF), colCom1(n), tbl(NULL), colCom2(NULL) {}
    TableConstraintDef(int rF, ColumnCommalist *n, Table *t) : rule_Flag(rF), colCom1(n), tbl(t), colCom2(NULL) {}
   // TableConstraintDef(int rule_Flag, SearchCondition *n2) : rule_Flag(rF), lit(NULL), srchCon(n2), tbl(NULL), colComList(NULL) {}
    TableConstraintDef(int rF, ColumnCommalist *n, Table *n2, ColumnCommalist *n3) : rule_Flag(rF), colCom1(n), tbl(n2), colCom2(n3) {}

	/**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // TABLE_CONSTRAINT_DEF_NODE_H
