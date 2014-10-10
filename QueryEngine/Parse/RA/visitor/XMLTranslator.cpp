#include "XMLTranslator.h"

#include "../ast/AggrExpr.h"
#include "../ast/AggrList.h"
#include "../ast/AntijoinOp.h"
#include "../ast/Attribute.h"
#include "../ast/AttrList.h"
#include "../ast/BinaryOp.h"
#include "../ast/Comparison.h"
#include "../ast/DiffOp.h"
#include "../ast/Expr.h"
#include "../ast/ExtendOp.h"
#include "../ast/GroupbyOp.h"
#include "../ast/JoinOp.h"
#include "../ast/MathExpr.h"
#include "../ast/OuterjoinOp.h"
#include "../ast/Predicate.h"
#include "../ast/ProductOp.h"
#include "../ast/Program.h"
#include "../ast/ProjectOp.h"
#include "../ast/Relation.h"
#include "../ast/RelExpr.h"
#include "../ast/RelExprList.h"
#include "../ast/RenameOp.h"
#include "../ast/ScanOp.h"
#include "../ast/SelectOp.h"
#include "../ast/SemijoinOp.h"
#include "../ast/SortOp.h"
#include "../ast/UnaryOp.h"
#include "../ast/UnionOp.h"

using namespace std;

namespace RA_Namespace {
    
    void XMLTranslator::visit(AggrExpr *v) {
        printTabs();
        tabCount_++;
        cout << "<AggrExpr func=\'" << v->func << "\'>" << endl;

        if (v->n1) v->n1->accept(*this);
        
        tabCount_--;
        printTabs();
        cout << "</AggrExpr>" << endl;
    }
    
    void XMLTranslator::visit(AggrList *v) {
        printTabs();
        tabCount_++;
        cout << "<AggrList>" << endl;
        
        if (v->n1) v->n1->accept(*this);
        if (v->n2) v->n2->accept(*this);
        
        tabCount_--;
        printTabs();
        cout << "</AggrList>" << endl;
    }
    
    void XMLTranslator::visit(AntijoinOp *v) {
        printTabs();
        tabCount_++;
        cout << "<AntijoinOp>" << endl;
        
        if (v->n1) v->n1->accept(*this);
        if (v->n2) v->n2->accept(*this);
        if (v->n3) v->n3->accept(*this);
        
        tabCount_--;
        printTabs();
        cout << "</AntijoinOp>" << endl;
    }
    
    void XMLTranslator::visit(Attribute *v) {
        printTabs();
        tabCount_++;
        cout << "<Attribute>";
        
        if (v->name1 != "" && v->name2 != "")
            cout << v->name1 << "." << v->name2 << endl;
        else
            cout << v->name1;
        
        cout << "</Attribute>" << endl;
        tabCount_--;
    }
    
    void XMLTranslator::visit(AttrList *v) {
        printTabs();
        tabCount_++;
        cout << "<AttrList>" << endl;
        
        if (v->n1) v->n1->accept(*this);
        if (v->n2) v->n2->accept(*this);
        
        tabCount_--;
        printTabs();
        cout << "</AttrList>" << endl;
    }
    
    void XMLTranslator::visit(Comparison *v) {
        printTabs();
        tabCount_++;
        cout << "<Comparison op=\'";
        
        if (v->op == OP_GT)
            cout << ">";
        else if (v->op == OP_GTE)
            cout << ">=";
        else if (v->op == OP_LT)
            cout << "<";
        else if (v->op == OP_LTE)
            cout << "<=";
        else if (v->op == OP_EQ)
            cout << "=";
        else if (v->op == OP_NEQ)
            cout << "!=";
        else
            cout << "?";
        
        cout << "\'" << ">" << endl;
        
        if (v->n1) v->n1->accept(*this);
        if (v->n2) v->n2->accept(*this);
        
        tabCount_--;
        printTabs();
        cout << "</Comparison>" << endl;
    }
    
    void XMLTranslator::visit(DiffOp *v) {
        printTabs();
        tabCount_++;
        cout << "<DiffOp>" << endl;
        
        if (v->n1) v->n1->accept(*this);
        if (v->n2) v->n2->accept(*this);
        if (v->n3) v->n3->accept(*this);
        
        tabCount_--;
        printTabs();
        cout << "</DiffOp>" << endl;
    }
    
    void XMLTranslator::visit(Expr *v) {
        printTabs();
        tabCount_++;
        cout << "<Expr>" << endl;
        
        if (v->n1) v->n1->accept(*this);
        if (v->n2) v->n2->accept(*this);
        
        tabCount_--;
        printTabs();
        cout << "</Expr>" << endl;
    }
    
    void XMLTranslator::visit(ExtendOp *v) {
        printTabs();
        tabCount_++;
        cout << "<ExtendOp name=\'" << v->name << "\'>" << endl;
        
        if (v->n1) v->n1->accept(*this);
        if (v->n2) v->n2->accept(*this);
        
        tabCount_--;
        printTabs();
        cout << "</ExtendOp>" << endl;
    }
    
    void XMLTranslator::visit(GroupbyOp *v) {
        printTabs();
        tabCount_++;
        cout << "<GroupbyOp>" << endl;
        
        if (v->n1) v->n1->accept(*this);
        if (v->n2) v->n2->accept(*this);
        if (v->n3) v->n3->accept(*this);
        
        tabCount_--;
        printTabs();
        cout << "</GroupbyOp>" << endl;
    }
    
    void XMLTranslator::visit(JoinOp *v) {
        printTabs();
        tabCount_++;
        cout << "<JoinOp>" << endl;
        
        if (v->n1) v->n1->accept(*this);
        if (v->n2) v->n2->accept(*this);
        if (v->n3) v->n3->accept(*this);
        
        tabCount_--;
        printTabs();
        cout << "</JoinOp>" << endl;
    }
    
    void XMLTranslator::visit(MathExpr *v) {
        if (v->isScalar) {
            printTabs();
            tabCount_++;
            cout << "<MathExpr>";
            if (v->intFloatFlag)
                cout << v->intVal;
            else
                cout << v->floatVal;
            cout << "</MathExpr>" << endl;
            tabCount_--;
            return;
        }
        else if (v->n1 && !v->n2) {
            v->n1->accept(*this);
        }
        else if (v->n1 && v->n2) {
            printTabs();
            tabCount_++;
            cout << "<MathExpr op='";
            if (v->op == OP_SUBTRACT)
                cout << "-";
            else if (v->op == OP_ADD)
                cout << "+";
            else if (v->op == OP_DIVIDE)
                cout << "/";
            else if (v->op == OP_MULTIPLY)
                cout << "*";
            else
                cout << "?";
            cout << "'>" << endl;
            
            if (v->n1) v->n1->accept(*this); // MathExpr
            if (v->n2) v->n2->accept(*this); // MathExpr
            
            tabCount_--;
            printTabs();
            cout << "</MathExpr>" << endl;
            return;
        }
        else if (v->n3) {
            printTabs();
            tabCount_++;
            cout << "<MathExpr>" << endl;
            v->n3->accept(*this); // Attribute
            tabCount_--;
            printTabs();
            cout << "</MathExpr>" << endl;
        }
        else if (v->n4) {
            printTabs();
            tabCount_++;
            cout << "<MathExpr>";
            v->n4->accept(*this);
            tabCount_--;
            printTabs();
            cout << "</MathExpr>" << endl;
        }
        
    }
    
    void XMLTranslator::visit(OuterjoinOp *v) {
        printTabs();
        tabCount_++;
        cout << "<OuterjoinOp>" << endl;
        
        if (v->n1) v->n1->accept(*this);
        if (v->n2) v->n2->accept(*this);
        if (v->n3) v->n3->accept(*this);
        
        tabCount_--;
        printTabs();
        cout << "</OuterjoinOp>" << endl;
    }
    
    void XMLTranslator::visit(Predicate *v) {
        printTabs();
        tabCount_++;
        if (v->op != "")
            cout << "<Predicate op='" << v->op << "'>" << endl;
        else
            cout << "<Predicate>" << endl;
        
        if (v->n1) v->n1->accept(*this);
        if (v->n2) v->n2->accept(*this);
        if (v->n3) v->n3->accept(*this);
        
        tabCount_--;
        printTabs();
        cout << "</Predicate>" << endl;
    }
    
    void XMLTranslator::visit(ProductOp *v) {
        printTabs();
        tabCount_++;
        cout << "<ProductOp>" << endl;
        
        if (v->n1) v->n1->accept(*this);
        if (v->n2) v->n2->accept(*this);
        
        tabCount_--;
        printTabs();
        cout << "</ProductOp>" << endl;
    }
    
    void XMLTranslator::visit(Program *v) {
        printTabs();
        tabCount_++;
        cout << "<Program>" << endl;
        
        if (v->n1) v->n1->accept(*this);
        
        tabCount_--;
        printTabs();
        cout << "</Program>" << endl;
    }
    
    void XMLTranslator::visit(ProjectOp *v) {
        printTabs();
        tabCount_++;
        cout << "<ProjectOp>" << endl;
        
        if (v->n1) v->n1->accept(*this);
        if (v->n2) v->n2->accept(*this);
        
        tabCount_--;
        printTabs();
        cout << "</ProjectOp>" << endl;
    }
    
    void XMLTranslator::visit(Relation *v) {
        printTabs();
        tabCount_++;
        cout << "<Relation>";
        cout << v->name;
        cout << "</Relation>" << endl;
        tabCount_--;
    }
    
    void XMLTranslator::visit(RelExpr *v) {
        printTabs();
        tabCount_++;
        cout << "<RelExpr>" << endl;
        
        if (v->n1) v->n1->accept(*this);
        if (v->n2) v->n2->accept(*this);
        if (v->n3) v->n3->accept(*this);
        if (v->n4) v->n4->accept(*this);
        
        tabCount_--;
        printTabs();
        cout << "</RelExpr>" << endl;
    }
    
    void XMLTranslator::visit(RelExprList *v) {
        printTabs();
        tabCount_++;
        cout << "<RelExprList>" << endl;
        
        if (v->n1) v->n1->accept(*this);
        if (v->n2) v->n2->accept(*this);
        
        tabCount_--;
        printTabs();
        cout << "</RelExprList>" << endl;
    }
    
    void XMLTranslator::visit(RenameOp *v) {
        printTabs();
        tabCount_++;
        cout << "<RenameOp>" << endl;
        
        if (v->n1) v->n1->accept(*this);

        printTabs();
        cout << "<old>" << v->name1 << "</old>" << endl;
        cout << "<new>" << v->name2 << "</new>" << endl;
        
        tabCount_--;
        printTabs();
        cout << "</RenameOp>" << endl;
    }
    
    void XMLTranslator::visit(ScanOp *v) {
        printTabs();
        tabCount_++;
        cout << "<ScanOp>" << endl;
        
        if (v->n1) v->n1->accept(*this);
        if (v->n2) v->n2->accept(*this);
        if (v->n3) v->n3->accept(*this);
        
        tabCount_--;
        printTabs();
        cout << "</ScanOp>" << endl;
    }
    
    void XMLTranslator::visit(SelectOp *v) {
        printTabs();
        tabCount_++;
        cout << "<SelectOp>" << endl;
        
        if (v->n1) v->n1->accept(*this);
        if (v->n2) v->n2->accept(*this);
        
        tabCount_--;
        printTabs();
        cout << "</SelectOp>" << endl;
    }
    
    void XMLTranslator::visit(SemijoinOp *v) {
        printTabs();
        tabCount_++;
        cout << "<SemijoinOp>" << endl;
        
        if (v->n1) v->n1->accept(*this);
        if (v->n2) v->n2->accept(*this);
        if (v->n3) v->n3->accept(*this);
        
        tabCount_--;
        printTabs();
        cout << "</SemijoinOp>" << endl;
    }
    
    void XMLTranslator::visit(SortOp *v) {
        printTabs();
        tabCount_++;
        cout << "<SortOp>" << endl;
        
        if (v->n1) v->n1->accept(*this);
        if (v->n2) v->n2->accept(*this);
        
        tabCount_--;
        printTabs();
        cout << "</SortOp>" << endl;
    }
    
    void XMLTranslator::visit(UnionOp *v) {
        printTabs();
        tabCount_++;
        cout << "<UnionOp>" << endl;
        
        if (v->n1) v->n1->accept(*this);
        if (v->n2) v->n2->accept(*this);
        
        tabCount_--;
        printTabs();
        cout << "</UnionOp>" << endl;
    }
    
} // RA_Namespace