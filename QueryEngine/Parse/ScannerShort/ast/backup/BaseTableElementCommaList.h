// forward class declarations
class BaseTableElement;
class BaseTableElementCommaList;

class BaseTableElementCommaList : public AST {
    BaseTableElement elem;
    BaseTableElementCommaList list;
    
    /**< Constructor */
    explicit BaseTableElementCommaList(BaseTableElement &n) : elem(n);
    BaseTableElementCommaList(BaseTableElementCommaList &n1, BaseTableElement &n2) : list(n1), elem(n2);
    
    /**< Accepts the given void visitor by calling v.visit(this) */
    virtual void accept(VoidVisitor &v) {
        v.visit(this);
    }
    
    /**< Passes a representation of this node to the outout stream */
    virtual void std::ostream& operator<<(std::ostream &stream, const A &a) {
        return strm << "<BaseTableElementCommaList>";
    }
}
