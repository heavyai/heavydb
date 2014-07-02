#ifndef AST_TABLE_H
#define AST_TABLE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

class Table : public ASTNode {
    
public:
    std::string name1;
    std::string name2;
    
    // if true, then "."; else "as"
    bool dotOrAs; 	// NAME '.' NAME
      				// NAME 'as' NAME
    
    /**< Constructor */
    explicit Table(const std::string &n1) : name1(n1) {
    	name2 = "";
    }

    Table(const std::string &n1, const std::string &tok, const std::string &n2) : name1(n1), name2(n2) {
    	if (tok == ".")
    		dotOrAs = true;
    }
    
    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // AST_TABLE_H
