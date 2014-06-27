#ifndef BASIC_H
#define BASIC_H

// forward class declarations
// class

class Basic : public ASTNode {
    
public:
    // SQLList &sqlList;
    
    /**< Constructor */
    // explicit Basic(SQLList &n) : sqlList(n);
    
    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(SimplePrinterVisitor &v) {
        v.visit(this);
    }
    
};

#endif // BASIC_H
