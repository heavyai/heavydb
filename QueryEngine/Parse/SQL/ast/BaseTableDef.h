#ifndef SQL_BASE_TABLE_DEF_NODE_H
#define SQL_BASE_TABLE_DEF_NODE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

namespace SQL_Namespace {
	class  BaseTableDef : public ASTNode {
    
public:
    std::string ddlCmd; // Should be CREATE or DROP
    Table *tbl;
    BaseTableElementCommalist *btec;
    
    /**< Constructor */
    BaseTableDef(const std::string &n1, Table *n2) : ddlCmd(n1), tbl(n2) {}
    BaseTableDef(const std::string &n1, Table *n2, BaseTableElementCommalist *n3)
        : ddlCmd(n1), tbl(n2), btec(n3) {}
    
    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
	};
}

#endif // SQL_BASE_TABLE_DEF_NODE_H
