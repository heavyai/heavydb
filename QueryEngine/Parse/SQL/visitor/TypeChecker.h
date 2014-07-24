/**
 * @file    TypeChecker.h
 * @author  Gil Walzer <gil@map-d.com>
 */
#ifndef SQL_TYPE_CHECK_VISITOR_H
#define SQL_TYPE_CHECK_VISITOR_H

#include "../visitor/Visitor.h"

#include "../ast/ASTNode.h"
#include "../ast/AbstractScalarExpr.h"
#include "../ast/ColumnCommalist.h"
#include "../ast/TableConstraintDef.h"
#include "../ast/BaseTableDef.h"
#include "../ast/BaseTableElementCommalist.h"
#include "../ast/BaseTableElement.h"
#include "../ast/Program.h"
#include "../ast/Table.h"
#include "../ast/Schema.h"
#include "../ast/SQL.h"
#include "../ast/ColumnDef.h"
#include "../ast/ColumnDefOpt.h"
#include "../ast/ColumnDefOptList.h"
#include "../ast/Column.h"
#include "../ast/Literal.h"
#include "../ast/DataType.h"
#include "../ast/SQLList.h"

#include "../ast/ManipulativeStatement.h"
#include "../ast/SelectStatement.h"
#include "../ast/Selection.h"
#include "../ast/OptAllDistinct.h"
#include "../ast/TableExp.h"
#include "../ast/FromClause.h"
#include "../ast/TableRefCommalist.h"
#include "../ast/TableRef.h"

#include "../ast/InsertStatement.h"
#include "../ast/OptColumnCommalist.h"
#include "../ast/ValuesOrQuerySpec.h"
#include "../ast/QuerySpec.h"
#include "../ast/InsertAtomCommalist.h"
#include "../ast/InsertAtom.h"
#include "../ast/Atom.h"

#include "../ast/SearchCondition.h"
#include "../ast/ScalarExpCommalist.h"
#include "../ast/ScalarExp.h"
#include "../ast/FunctionRef.h"
#include "../ast/Ammsc.h"
#include "../ast/Predicate.h"
#include "../ast/ComparisonPredicate.h"
#include "../ast/BetweenPredicate.h"
#include "../ast/LikePredicate.h"
#include "../ast/OptEscape.h"
#include "../ast/ColumnRef.h"

#include "../ast/ColumnRefCommalist.h"
#include "../ast/OptWhereClause.h"
#include "../ast/OptGroupByClause.h"
#include "../ast/OptHavingClause.h"
#include "../ast/OptLimitClause.h"
#include "../ast/OptAscDesc.h"
#include "../ast/OrderingSpecCommalist.h"
#include "../ast/OrderingSpec.h"
#include "../ast/OptOrderByClause.h"

#include "../ast/UpdateStatementSearched.h"
#include "../ast/UpdateStatementPositioned.h"
#include "../ast/AssignmentCommalist.h"
#include "../ast/Assignment.h"
#include "../ast/Cursor.h"

#include "../ast/TestForNull.h"
#include "../ast/InPredicate.h"
#include "../ast/ExistenceTest.h"
#include "../ast/AllOrAnyPredicate.h"
#include "../ast/AnyAllSome.h"
#include "../ast/AtomCommalist.h"
#include "../ast/Subquery.h"
#include "../ast/GroupByList.h"

#include <iostream>
using std::cout;
using std::endl;

#define TAB_SIZE 2 // number of spaces in a tab

namespace SQL_Namespace {

enum tabFlag {INCR, DECR, NONE};
/**
 * @todo brief and detailed descriptions
 */
class TypeChecker : public Visitor {

public:

    void visit(class BaseTableDef *v) {
    }

    void visit(class TableConstraintDef *v) {
    }

    void visit(class ColumnDefOpt *v) {
    }

    void visit(class ColumnDefOptList *v) {
    
        if (v->colDefOptList) v->colDefOptList->accept(*this);
        v->colDefOpt->accept(*this);

    }
    
    void visit(class ColumnDef *v) {
        
        v->col->accept(*this);
        v->dType->accept(*this);
        if (v->colDefOptList) v->colDefOptList->accept(*this);
    }
    void visit(class InPredicate *v) {
    
        v->se->accept(*this);
    
        if (v->sq) v->sq->accept(*this);
        if (v->ac) v->ac->accept(*this);

    }
    
    void visit(class TestForNull *v) {
    
        v->cr->accept(*this);
    
    }
    void visit(class AllOrAnyPredicate *v) {
        v->se->accept(*this);
        v->aas->accept(*this);
        v->sq->accept(*this);
    }
    void visit(class Subquery *v) {
        v->oad->accept(*this);
        v->s->accept(*this);
        v->te->accept(*this);
    }

    void visit(class AnyAllSome *v) {
    }

    void visit(class ExistenceTest *v) {

        v->sq->accept(*this);
    }

    void visit(class BaseTableElement *v) {
        if (v->colDef) v->colDef->accept(*this);
        if (v->tblConDef) v->tblConDef->accept(*this);
    }
    
    void visit(class BaseTableElementCommalist *v) {
        if (v->btec) v->btec->accept(*this);
        v->bte->accept(*this);

    }

    void visit(class Assignment *v) {
        
        v->c->accept(*this);
        if (v->se) v->se->accept(*this);
    }

    void visit(class AssignmentCommalist *v) {
  
        if (v->ac) v->ac->accept(*this);
        v->a->accept(*this);

    }

    void visit(class OrderingSpec *v) {
        
        if (v->cr) v->cr->accept(*this);
        if (v->oad) v->oad->accept(*this);
    }

    void visit(class OrderingSpecCommalist *v) {
  
        if (v->osc) v->osc->accept(*this);
        v->os->accept(*this);

    }

    void visit(class OptOrderByClause *v) {    
        v->osc->accept(*this);
    }

    void visit(class OptLimitClause *v) {
    }

    void visit(class OptHavingClause *v) {
        v->sc->accept(*this);
    }
    
    void visit(class OptWhereClause *v) {
        v->sc->accept(*this);
    }
    
    void visit(class OptGroupByClause *v) {
        //if (v->crc) v->crc->accept(*this);
        if (v->gbl) v->gbl->accept(*this);
    }
    
    void visit(class GroupByList *v) {
    
        if (v->gbl) v->gbl->accept(*this);
        v->se->accept(*this);
        v->oad->accept(*this);
    }

    void visit(class ColumnCommalist *v) {
        if (v->colCom) v->colCom->accept(*this);
        v->col->accept(*this);
    }

    void visit(class OptColumnCommalist *v) {

        if (v->cc) v->cc->accept(*this);
    }

    void visit(class Atom *v) {

        if (v->lit) {
            v->lit->accept(*this);
            //printf("type is: %d\n", v->lit->getType());
            v->setType(v->lit->getType());
            v->setColno(v->lit->getColno());
            v->setLineno(v->lit->getLineno());
        }
//        if (v->user == "user") cout << "<USER>" << endl;

    }

    void visit(class AtomCommalist *v) {

        if (v->ac) v->ac->accept(*this);
        v->a->accept(*this);

    }

    void visit(class InsertAtom *v) {

        if (v->a) v->a->accept(*this);
    }

    void visit(class InsertAtomCommalist *v) {

        if (v->iac) v->iac->accept(*this);
        v->ia->accept(*this);
    }

    void visit(class QuerySpec *v) {
        v->OAD->accept(*this);
        v->sel->accept(*this);
        v->tblExp->accept(*this);
    }

    void visit(class ValuesOrQuerySpec *v) {

        if (v->iac) v->iac->accept(*this);
        if (v->qs) v->qs->accept(*this);
    }

    void visit(class UpdateStatementSearched *v) {

        v->tbl->accept(*this);
        v->ac->accept(*this);
        v->owc->accept(*this);
    }

    void visit(class UpdateStatementPositioned *v) {
        v->tbl->accept(*this);
        v->ac->accept(*this);
        v->c->accept(*this);
    }

    void visit(class InsertStatement *v) {
        v->tbl->accept(*this);
        v->oCC->accept(*this);
        v->voQS->accept(*this);
    }

    void visit(class TableRef *v) {
        if (v->tbl) v->tbl->accept(*this);
    }
    
    void visit(class TableRefCommalist *v) {
        if (v->trc) v->trc->accept(*this);
        v->tr->accept(*this);
    }

    void visit(class FromClause *v) {
        if (v->trc) {
            v->trc->accept(*this);
        }
        if (v->ss) v->ss->accept(*this);
    }
    
    void visit(TableExp *v) {
        v->fc->accept(*this);
        if (v->owc) v->owc->accept(*this);
        if (v->ogbc) v->ogbc->accept(*this);
        if (v->ohc) v->ohc->accept(*this);
        if (v->oobc) v->oobc->accept(*this);
        if (v->olc) v->olc->accept(*this);
    }

    void visit(class Selection *v) {

        if(v->sec) {    
            v->sec->accept(*this);
        }
    }

    void visit(class OptEscape *v) {
        v->a->accept(*this);
    }

    void visit(class ScalarExpCommalist *v) {

        if (v->sec) v->sec->accept(*this);
        if (v->se) {
            v->se->accept(*this);
           // cout << "Final type of this scalar exp: " << v->se->getType() << endl;
        }
    }

    void visit(class ScalarExp *v) {
        /* rules are:
        0 (scalar_exp)
        1 addition
        2 subtraction
        3 multiplication
        4 division
        5 positive [scalar_exp]
        6 negative [scalar_exp] */
        
        //cout << v->rule_Flag << endl;
        // check the types with only one scalar expression 
        if (v->rule_Flag == 0 || v->rule_Flag == 5 || v->rule_Flag == 6) {
            v->se1->accept(*this);
            v->setType(v->se1->getType());
            
            // set position of higher scalar expression to be that of child
            v->setLineno(v->se1->getLineno());
            v->setColno(v->se1->getColno());
        }

        // make sure the types match
        else if (1 <= v->rule_Flag && v->rule_Flag <= 4) {
            v->se1->accept(*this);
            v->se2->accept(*this);
            
            //being clever to avoid bloat: check if exactly one type is a string
            if ((v->se1->getType() == SCALAR_STRING) ^ (v->se2->getType() == SCALAR_STRING)) {
                printf("scalar expression mismatch between types, line %d, column %d\n", 
                    v->se1->getLineno(), v->se1->getColno());
                }

            // find common type; if INT and FLOAT, cast to higher of two
            else {
         //       cout << "pigs (two different ones)" << endl;
                // if one is a string, both must be strings;
                if (v->se1->getType() == SCALAR_STRING) v->setType(SCALAR_STRING);
                else if (v->se1->getType() == SCALAR_INT) {
                    // if two ints...
                    if (v->se2->getType() == SCALAR_INT) v->setType(SCALAR_INT);
                    // if at least one float...
                    else v->setType(SCALAR_FLOAT);
                }
            }
            //set position of scalar expression to be position of first sub expression
            v->setLineno(v->se1->getLineno());
            v->setColno(v->se1->getColno());
        }

        // not a math expression
        else {
            if (v->a) {
           //     cout << "it's an atom\n";
                v->a->accept(*this);
                v->setType(v->a->getType());

                v->setLineno(v->a->getLineno());
                v->setColno(v->a->getColno());            
            }
            if (v->cr) {
                v->cr->accept(*this);
                v->setType(v->cr->getType());

                v->setLineno(v->cr->getLineno());
                v->setColno(v->cr->getColno());
            }
            if (v->fr) {
                v->fr->accept(*this);
                v->setType(v->fr->getType());

                v->setLineno(v->fr->getLineno());
                v->setColno(v->fr->getColno());
            } 
        }
    }

    void visit(class SearchCondition *v) {
        if (v->p) v->p->accept(*this);
    }
    
    void visit(class LikePredicate *v) {
    }

    void visit(class ComparisonPredicate *v) {
        if (v->se1) {
            v->se1->accept(*this);
            //cout << "Final type of this scalar exp: " << v->se2->getType() << endl;
        }
        if (v->se2) {
            v->se2->accept(*this);
            //cout << "Final type of this scalar exp: " << v->se2->getType() << endl;
        }
        if (v->s) v->s->accept(*this);

    }

    void visit(class BetweenPredicate *v) {

        // strings can be between strings, numerics between numerics
        v->se1->accept(*this);
        ScalarExprType type1 = v->se1->getType();

        v->se2->accept(*this);
        ScalarExprType type2 = v->se2->getType();

        v->se3->accept(*this);
        ScalarExprType type3 = v->se3->getType();

        //based on the type of the sandwiched scalar expression, check types of predicate
        if (type1 == SCALAR_STRING) {
            if (type2 != SCALAR_STRING) {
                // placeholders for line and column number, fill in later 
                printf("BETWEEN predicate expects string: mismatch on line %d, column %d\n",
                    v->se1->getLineno(), v->se1->getColno());
                exit(0);
            }
            if (type3 != SCALAR_STRING) {
                // placeholders for line and column number, fill in later 
                printf("BETWEEN predicate expects string: mismatch on line %d, column %d\n",
                    v->se1->getLineno(), v->se1->getColno());
                exit(0);
            }
        }
        
    }

    void visit(class Predicate *v) {
        if (v->cp) v->cp->accept(*this);
        if (v->bp) v->bp->accept(*this);
        if (v->lp) v->lp->accept(*this);
    }

    void visit(class FunctionRef *v) {

        // for these functions, return type is fixed; no need to visit arguments
        if (v->func_name == "count") v->setType(SCALAR_INT);
        else if (v->func_name == "substr") v->setType(SCALAR_STRING); 
        
        // for other functions, return type is the same as the argument type
        else {
            // if func(column1.column2) visit columnRef- catalog will assign type  
            if (v->cr) {
                v->cr->accept(*this);

                // the argument of the function
                v->setType(v->cr->getType());
                v->setLineno(v->cr->getLineno());
                v->setColno(v->cr->getColno());
            }

            // if func(scalar_exp) visit scalarExp- type will be assigned during (accept)
            if (v->se1) {
                v->se1->accept(*this);
                v->setType(v->se1->getType());
                v->setLineno(v->se1->getLineno());
                v->setColno(v->se1->getColno());
            }
        }


        // check for specific return types of functions
    }
    
    void visit(class OptAllDistinct *v) {
    }
    
    void visit(class SelectStatement *v) {
        if (v->OAD) v->OAD->accept(*this);
        if (v->sel) v->sel->accept(*this);
        if (v->tblExp) v->tblExp->accept(*this);
    }
    
    void visit(class ManipulativeStatement *v) {
        if (v->selSta) v->selSta->accept(*this);
        if (v->USP) v->USP->accept(*this);
        if (v->USS) v->USS->accept(*this);
        if (v->inSta) v->inSta->accept(*this); 
    }

    void visit(class Schema *v) {
        if (v->btd) v->btd->accept(*this);
    }
    
    void visit(class SQL *v) {        
        if (v->sch) v->sch->accept(*this);
        if (v->manSta) v->manSta->accept(*this);
    }

    void visit(class SQLList *v) {
        if (v->sqlList) { v->sqlList->accept(*this);  }
        v->sql->accept(*this);
    }


    void visit(class Program *v) {
        if (v->sqlList) v->sqlList->accept(*this);
    }

    void visit(class DataType *v) {
    }

    void visit(class Literal *v) {
   //     if (v->name1 != "") cout << "<Literal string='" << v->name1 << "'/>" << endl;
   //     if (v->int1 != 0) cout << "<Literal int = " << v->int1 << "/>" << endl;
    }

    void visit(class ColumnRef *v) {

        //Catalog should check types!
        
    }

    void visit(class ColumnRefCommalist *v) {
        if (v->crc) v->crc->accept(*this);
        v->cr->accept(*this);
    }

    void visit(class Cursor *v) {
    }

    void visit(class Column *v) {
    }

    void visit(class OptAscDesc *v) {
    }

    void visit(class Ammsc *v) {
    }

    void visit(class Table *v) {

    }

private:
    static int tabLevel_;   /**< Keeps track of number of tabs to print out on a line. */
};

} // SQL_Namespace

#endif // SQL_TYPE_CHECK_VISITOR_H
