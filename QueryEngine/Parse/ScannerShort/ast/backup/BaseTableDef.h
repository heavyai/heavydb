// forward class declarations
class Table;

class BaseTableDef : public AST {
    Table table;
    
    /**< Constructor */
    explicit BaseTableDef(Table &n) : table(n);
    
    /**< Accepts the given void visitor by calling v.visit(this) */
    virtual void accept(VoidVisitor &v) {
        v.visit(this);
    }
    
    /**< Passes a representation of this node to the outout stream */
    virtual void std::ostream& operator<<(std::ostream &stream, const A &a) {
        return strm << "<BaseTableDef>";
    }
}
