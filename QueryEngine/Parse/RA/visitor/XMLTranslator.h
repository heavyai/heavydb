/**
 * @file    XMLTranslator.h
 * @author  Steven Stewart <steve@map-d.com>
 */
#ifndef QueryEngine_Parse_RA_Visitor_XMLTranslator_h
#define QueryEngine_Parse_RA_Visitor_XMLTranslator_h

#include <iostream>
#include "Visitor.h"

namespace RA_Namespace {
    
    class XMLTranslator : public RA_Namespace::Visitor {

    public:
        
        /// Constructor
        XMLTranslator() {}
        
        /// Destructor
        ~XMLTranslator() {}
        
        virtual void visit(AggrExpr *v);
        virtual void visit(AggrList *v);
        virtual void visit(AntijoinOp *v);
        virtual void visit(Attribute *v);
        virtual void visit(AttrList *v);
        virtual void visit(Comparison *v);
        virtual void visit(DiffOp *v);
        virtual void visit(Expr *v);
        virtual void visit(ExtendOp *v);
        virtual void visit(GroupbyOp *v);
        virtual void visit(JoinOp *v);
        virtual void visit(MathExpr *v);
        virtual void visit(OuterjoinOp *v);
        virtual void visit(Predicate *v);
        virtual void visit(ProductOp *v);
        virtual void visit(Program *v);
        virtual void visit(ProjectOp *v);
        virtual void visit(Relation *v);
        virtual void visit(RelExpr *v);
        virtual void visit(RelExprList *v);
        virtual void visit(RenameOp *v);
        virtual void visit(ScanOp *v);
        virtual void visit(SelectOp *v);
        virtual void visit(SemijoinOp *v);
        virtual void visit(SortOp *v);
        virtual void visit(UnionOp *v);
        
    private:
        int tabCount_ = 0;
        
        inline void printTabs() {
            for (int i = 0; i < tabCount_; ++i)
                std::cout << "   ";
        }
    };
    
} // RA_Namespace

#endif // QueryEngine_Parse_RA_Visitor_XMLTranslator_h
