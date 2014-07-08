#ifndef AST_TABLE_H
#define AST_TABLE_H

#include "ASTNode.h"
#include "../visitor/Visitor.h"

class Table : public ASTNode {
    
public:
    std::string name1;
    std::string name2;
    std::string name3;

    // if true, then "."; else "as"
    bool dotOrAs; 	// NAME '.' NAME
      				// NAME 'as' NAME
    

    int rule_Flag;
    /* Rules:
    0 '.'
    1 AS */

    /**< Constructor */
    explicit Table(const std::string &n1) : name1(n1) {
    	name2 = "";
    }

    Table(int rF, const std::string &n1, const std::string &n2) : rule_Flag(rF), name1(n1), name2(n2) {
     }

    Table(const std::string &n1, const std::string &n2, const std::string &n3) : name1(n1), name3(n3) {
     		dotOrAs = true;
    }
    
    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // AST_TABLE_H
