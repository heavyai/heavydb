#ifndef AST_SIMPLE_PRINTER_VISITOR_H
#define AST_SIMPLE_PRINTER_VISITOR_H

#include "Visitor.h"

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

#include <iostream>
using std::cout;
using std::endl;

#define TAB_SIZE 2 // number of spaces in a tab

enum tabFlag {INCR, DECR, NONE};
/**
 * @todo brief and detailed descriptions
 */
class XMLTranslator : public Visitor {

public:

    static void printTabs(tabFlag flag) {
        if (flag == INCR)
            tabLevel_++;
        //cout << tabLevel_;
        for (int i = 0; i < tabLevel_; ++i)
            for (int j = 0; j < TAB_SIZE; ++j)
                cout << " ";
        if (flag == DECR)
            tabLevel_--;
    }

    void visit(class BaseTableDef *v) {
        printTabs(INCR);
        cout << "<BaseTableDef ddlCmd='" << v->ddlCmd << "'>" << endl;

        if (v->ddlCmd == "CREATE") {
            if (!v->tbl || !v->btec) {
                printTabs(INCR);
                cout << "<ERROR msg='Invalid parameters to ddlCmd: " << v->ddlCmd << "'/>" << endl;
                tabLevel_--;
            }
            else {
                v->tbl->accept(*this);

                printTabs(INCR);
                cout << "</BaseTableElementCommalist>" << endl;
          
                v->btec->accept(*this);
                
                printTabs(DECR);
                cout << "</BaseTableElementCommalist>" << endl;
            }
        }
        else if (v->ddlCmd == "DROP") {
            if (!v->tbl) {
                printTabs(INCR);
                cout << "<ERROR msg='Invalid parameters to ddlCmd: " << v->ddlCmd << "'>" << endl;
                printTabs(DECR);
            }
            else
                v->tbl->accept(*this);
        }
        else {
            printTabs(INCR);
            cout << "<ERROR msg='Invalid ddlCmd: " << v->ddlCmd << "'>" << endl;
            printTabs(DECR);
        }

        printTabs(DECR);
        cout << "</BaseTableDef>" << endl;
    }

    void visit(class TableConstraintDef *v) {
        printTabs(INCR);
        cout << "<colDefOpt>" << endl;

         /* rule_Flags: 
        0 NOT NULL
        1 NOT NULL PRIMARY KEY
        2 NOT NULL UNIQUE
        3 DEFAULT [literal]
        4 DEFAULT NULL
        5 DEFAULT USER
        6 CHECK [search condition]
        7 REFERENCES [table]
        8 REFERENCES [table] [column_commalist]
        */

        int rf = v->rule_Flag;
        if (rf == 0) { // Handle UNIQUE column_commalist
            printTabs(INCR);
            cout << "<UNIQUE>" << endl;

            v->colCom1->accept(*this);
            
            cout << "</>" << endl;
            printTabs(DECR);
        }
        else if (rf == 1) { // Handle PRIMARY KEY column_commalist
            printTabs(INCR);
            cout << "<PRIMARY KEY>" << endl;

            v->colCom1->accept(*this);

            cout << "</>" << endl;
            printTabs(DECR);
        }
        else if (rf == 2) { // Handle FOREIGN KEY column_commalist REFERENCES table [possibly] (column_commalist)
            printTabs(INCR);
            cout << "<FOREIGN KEY>" << endl;

            v->colCom1->accept(*this);
            
            cout << "</>" << endl << "<REFERENCES>" << endl;
            
            v->tbl->accept(*this);

            cout << "</>" << endl;

            if(v->colCom2) {
                v->colCom2->accept(*this);
            }

            printTabs(DECR);
            cout << "</BaseTableDef>" << endl;
        }/*
        else if (rf == 3) { // Handle CHECK search_condition
            printTabs(INCR);
            cout << "<DEFAULT>" << endl;
            v->srchCon->accept(*this);
            cout << "</>" << endl;
            printTabs(DECR);
        } */
       cout << "</TableConstraintDef>" << endl;
    }

    void visit(class ColumnDefOpt *v) {
        printTabs(INCR);
        cout << "<colDefOpt>" << endl;

         /* rule_Flags: 
        0 NOT NULL
        1 NOT NULL PRIMARY KEY
        2 NOT NULL UNIQUE
        3 DEFAULT [literal]
        4 DEFAULT NULL
        5 DEFAULT USER
        6 CHECK [search condition]
        7 REFERENCES [table]
        8 REFERENCES [table] [column_commalist]
        */

        int rf = v->rule_Flag;
        if (rf == 0) { // Handle NOT NULL
            printTabs(INCR);
            cout << "<NOT NULL> </>" << endl;
            printTabs(DECR);
        }
        else if (rf == 1) { // Handle NOT NULL PRIMARY KEY
            printTabs(INCR);
            cout << "<NOT NULL PRIMARY KEY> </>" << endl;
            printTabs(DECR);
        }
        else if (rf == 2) { // Handle NOT NULL UNIQUE
            printTabs(INCR);
            cout << "<NOT NULL UNIQUE> </>" << endl;
            printTabs(DECR);
        }
        else if (rf == 3) { // Handle DEFAULT literal
            printTabs(INCR);
            cout << "<DEFAULT>" << endl;
            v->lit->accept(*this);
            cout << "</>" << endl;
            printTabs(DECR);
        } 
        else if (rf == 4) { // Handle DEFAULT NULL
            printTabs(INCR);
            cout << "<DEFAULT NULL> </>" << endl;
            printTabs(DECR);
        }
        else if (rf == 5) { // Handle DEFAULT USER
            printTabs(INCR);
            cout << "<DEFAULT USER> </>" << endl;
            printTabs(DECR);
        } 
        else if (rf == 6) { // Handle CHECK search_condition
            printTabs(INCR);
            cout << "<CHECK>" << endl;
            v->srchCon->accept(*this);
            cout << "</>" << endl;
            printTabs(DECR);
        } 
        else if (rf == 7) { // Handle REFERENCES table
            printTabs(INCR);
            cout << "<REFERENCES>" << endl;
            v->tbl->accept(*this);
            cout << "</>" << endl;
            printTabs(DECR);
        } 
        else if (rf == 8) { // Handle REFERENCES table commalist 
            printTabs(INCR);
            cout << "<REFERENCES>" << endl;
            v->tbl->accept(*this);
            v->colComList->accept(*this);
            cout << "</>" << endl;
            printTabs(DECR);
        }

        printTabs(DECR);
        cout << "</ColDefOpt>" << endl;
    }

    void visit(class ColumnDefOptList *v) {
        printTabs(INCR);
        cout << "<ColumnDefOptList>" << endl;

        if (v->colDefOptList) v->colDefOptList->accept(*this);
        v->colDefOpt->accept(*this);

        printTabs(DECR);
        cout << "</ColumnDefOptList>" << endl;
    }
    
    void visit(class ColumnDef *v) {
        printTabs(INCR);
        cout << "<ColumnDef>" << endl;
        
        v->col->accept(*this);
        v->dType->accept(*this);
        if (v->colDefOptList) v->colDefOptList->accept(*this);
        
        printTabs(DECR);
        cout << "</ColumnDef>" << endl;
    }
      void visit(class InPredicate *v) {
        printTabs(INCR);
        cout << "<InPredicate>" << endl;

        v->se->accept(*this);
        printTabs(NONE);
        if (v->rule_Flag == 0) cout << "<NOT IN>" << endl;
        else cout << "<NOT>" << endl;

        if (v->sq) v->sq->accept(*this);
        if (v->ac) v->ac->accept(*this);

        printTabs(DECR);
        cout << "</InPredicate>" << endl;
    }
    
      void visit(class TestForNull *v) {
        printTabs(INCR);
        cout << "<TestForNull>" << endl;

        v->cr->accept(*this);
        printTabs(NONE);
        if (v->rule_Flag == 0) cout << "<NOT IN>" << endl;
        else cout << "<NOT>" << endl;

        printTabs(DECR);
        cout << "</TestForNull>" << endl;
    }
    void visit(class AllOrAnyPredicate *v) {
        printTabs(INCR);
        cout << "<AllOrAnyPredicate>" << endl;

        v->se->accept(*this);
        v->aas->accept(*this);
        v->sq->accept(*this);

        printTabs(DECR);
        cout << "</AllOrAnyPredicate>" << endl;
    }
    void visit(class Subquery *v) {
        printTabs(INCR);
        cout << "<Subquery>" << endl;

        v->oad->accept(*this);
        v->s->accept(*this);
        v->te->accept(*this);

        printTabs(DECR);
        cout << "</Subquery>" << endl;
    }

    void visit(class AnyAllSome *v) {
        printTabs(INCR);

        cout << "<AnyAllSome value= " << v->anyAllSome << "/>" << endl;
        
        printTabs(DECR);
        cout << "</AnyAllSome>" << endl;
    }

    void visit(class ExistenceTest *v) {
        printTabs(INCR);
        cout << "<ExistenceTest>" << endl;
    
        printTabs(NONE);
        cout << "<EXISTS>" << endl;
        v->sq->accept(*this);

        printTabs(DECR);
        cout << "</ExistenceTest>" << endl;
    }

    void visit(class BaseTableElement *v) {
        printTabs(INCR);
        cout << "<BaseTableElement>" << endl;

        if (v->colDef) v->colDef->accept(*this);
        if (v->tblConDef) v->tblConDef->accept(*this);

        printTabs(DECR);
        cout << "</BaseTableElement>" << endl;
    }
    
    void visit(class BaseTableElementCommalist *v) {
        printTabs(INCR);
        cout << "<BaseTableElementCommalist>" << endl;

        if (v->btec) v->btec->accept(*this);
        v->bte->accept(*this);

    }

    void visit(class Assignment *v) {
        printTabs(INCR);
        cout << "<Assignment>" << endl;

        v->c->accept(*this);
        printTabs(NONE);
        cout << "\t<ASSIGNMENT>" << endl;
        if (v->se) v->se->accept(*this);
        else {
            printTabs(NONE);
            cout << "NULL" << endl;
        }
        printTabs(DECR);
        cout << "<Assignment>" << endl;
    }

    void visit(class AssignmentCommalist *v) {
  
        if (v->ac) v->ac->accept(*this);
        v->a->accept(*this);

    }

    void visit(class OrderingSpec *v) {
        printTabs(INCR);
        cout << "<OrderingSpec>" << endl;

        if (v->cr) v->cr->accept(*this);
        else {
            printTabs(NONE);
            cout << "<" << v->orderInt << ">" << endl;
        }
        if (v->oad) v->oad->accept(*this);

        printTabs(DECR);
        cout << "<OrderingSpec>" << endl;
    }

    void visit(class OrderingSpecCommalist *v) {
  
        if (v->osc) v->osc->accept(*this);
        v->os->accept(*this);

    }

    void visit(class OptOrderByClause *v) {
        printTabs(INCR);
        cout << "<OptOrderByClause>" << endl;

        printTabs(INCR);
        cout << "<OptOrderingSpecCommalist>" << endl;
        
        v->osc->accept(*this);
        
        printTabs(DECR);
        cout << "<OptOrderingSpecCommalist>" << endl;
        
        printTabs(DECR);
        cout << "</OptOrderByClause>" << endl;
    }

    void visit(class OptLimitClause *v) {
        printTabs(INCR);
        if (v->rule_Flag == 0)
            cout << "<OptLimitClause limit = " << v->lim1 << "," << v->lim2 << " >" << endl;
        else if (v->rule_Flag == 1)
            cout << "<OptLimitClause limit = " << v->lim1 << " OFFSET " << v->lim2 << " >" << endl;
        else
            cout << "<OptLimitClause limit = " << v->lim1 << ">" << endl;

        printTabs(DECR);
        cout << "</OptLimitClause>" << endl;
    }

    void visit(class OptHavingClause *v) {
        printTabs(INCR);
        cout << "<OptHavingClause>" << endl;

        v->sc->accept(*this);

        printTabs(DECR);
        cout << "</OptHavingClause>" << endl;
    }
    
    void visit(class OptWhereClause *v) {
        printTabs(INCR);
        cout << "<OptWhereClause>" << endl;

        v->sc->accept(*this);

        printTabs(DECR);
        cout << "</OptWhereClause>" << endl;
    }
    
    void visit(class OptGroupByClause *v) {
        printTabs(INCR);
        cout << "<OptGroupByClause>" << endl;

        printTabs(INCR);
        cout << "<ColumnRefCommalist>" << endl;

        v->crc->accept(*this);

        printTabs(DECR);
        cout << "</ColumnRefCommalist>" << endl;

        printTabs(DECR);
        cout << "</OptGroupByClause>" << endl;
    }
    
    void visit(class ColumnCommalist *v) {
        if (v->colCom) v->colCom->accept(*this);
        v->col->accept(*this);
    }

    void visit(class OptColumnCommalist *v) {
        printTabs(INCR);
        cout << "<OptColumnCommalist>" << endl;

        printTabs(INCR);
        cout << "<ColumnCommalist>" << endl;

        if (v->cc) v->cc->accept(*this);


        printTabs(DECR);
        cout << "</ColumnCommalist>" << endl;

        printTabs(DECR);
        cout << "</OptColumnCommalist>" << endl;
    }

    void visit(class Atom *v) {
        printTabs(INCR);
        cout << "<Atom>" << endl;

        if (v->lit) v->lit->accept(*this);
        if (v->user == "user") cout << "<USER>" << endl;

        printTabs(DECR);
        cout << "</Atom>" << endl;
    }

    void visit(class AtomCommalist *v) {

        if (v->ac) v->ac->accept(*this);
        v->a->accept(*this);

    }

    void visit(class InsertAtom *v) {
        printTabs(INCR);
        cout << "<InsertAtom>" << endl;

        if (v->a) v->a->accept(*this);
        
        printTabs(DECR);
        cout << "</InsertAtom>" << endl;
    }

    void visit(class InsertAtomCommalist *v) {

        if (v->iac) v->iac->accept(*this);
        v->ia->accept(*this);
    }

    void visit(class QuerySpec *v) {
        printTabs(INCR);
        cout << "<QuerySpec>" << endl;

        v->OAD->accept(*this);
        v->sel->accept(*this);
        v->tblExp->accept(*this);

        printTabs(DECR);
        cout << "</QuerySpec>" << endl;
    }

    void visit(class ValuesOrQuerySpec *v) {
        printTabs(INCR);
        cout << "<ValuesOrQuerySpec>" << endl;

        if (v->iac) v->iac->accept(*this);
        if (v->qs) v->qs->accept(*this);

        printTabs(DECR);
        cout << "</ValuesOrQuerySpec>" << endl;
    }

    void visit(class UpdateStatementSearched *v) {
        printTabs(INCR);
        cout << "<UpdateStatementSearched>" << endl;

        v->tbl->accept(*this);

        printTabs(INCR);
        cout << "<AtomCommalist>" << endl;
        
        v->ac->accept(*this);
        
        printTabs(DECR);
        cout << "<AtomCommalist>" << endl;

        v->owc->accept(*this);

        printTabs(DECR);
        cout << "</UpdateStatementSearched>" << endl;
    }

    void visit(class UpdateStatementPositioned *v) {
        printTabs(INCR);
        cout << "<UpdateStatementPositioned>" << endl;

        v->tbl->accept(*this);
        v->ac->accept(*this);
        v->c->accept(*this);

        printTabs(DECR);
        cout << "</UpdateStatementPositioned>" << endl;
    }

    void visit(class InsertStatement *v) {
        printTabs(INCR);
        cout << "<InsertStatement>" << endl;

        v->tbl->accept(*this);
        v->oCC->accept(*this);
        v->voQS->accept(*this);

        printTabs(DECR);
        cout << "</InsertStatement>" << endl;
    }

    void visit(class TableRef *v) {
        printTabs(INCR);
        cout << "<TableRef>" << endl;

        if (v->tbl) v->tbl->accept(*this);

        printTabs(DECR);
        cout << "</TableRef>" << endl;
    }
    
    void visit(class TableRefCommalist *v) {
        if (v->trc) v->trc->accept(*this);
        v->tr->accept(*this);
    }

    void visit(class FromClause *v) {
        printTabs(INCR);
        cout << "<FromClause>" << endl;

        if (v->trc) {
            printTabs(INCR);
            cout << "<TableRefCommalist>" << endl;

            v->trc->accept(*this);

            printTabs(DECR);
            cout << "</TableRefCommalist>" << endl;
        }
        if (v->ss) v->ss->accept(*this);

        printTabs(DECR);
        cout << "</FromClause>" << endl;
    }
    
    void visit(TableExp *v) {
        printTabs(INCR);
        cout << "<TableExp>" << endl;

        v->fc->accept(*this);
        if (v->owc) v->owc->accept(*this);
        if (v->ogbc) v->ogbc->accept(*this);
        if (v->ohc) v->ohc->accept(*this);
        if (v->oobc) v->oobc->accept(*this);
        if (v->olc) v->olc->accept(*this);

        printTabs(DECR);
        cout << "</TableExp>" << endl;
    }

    void visit(class Selection *v) {
        printTabs(INCR);
        cout << "<Selection>" << endl;

        if(v->sec) {
            printTabs(INCR);
            cout << "<ScalarExpCommalist>" << endl;
            
            v->sec->accept(*this);

            printTabs(DECR);
             cout << "</ScalarExpCommalist>" << endl;
        }
        else {
            printTabs(NONE);
            cout << "\t<*>" << endl;
        }

        printTabs(DECR);
        cout << "</Selection>" << endl;
    }

    void visit(class OptEscape *v) {
        printTabs(INCR);
        cout << "<OptEscape>" << endl;

        v->a->accept(*this);

        printTabs(DECR);
        cout << "</OptEscape>" << endl;
    }

    void visit(class ScalarExpCommalist *v) {

        if (v->sec) v->sec->accept(*this);
        v->se->accept(*this);
    }

    void visit(class ScalarExp *v) {
        printTabs(INCR);
        cout << "<ScalarExp>" << endl;
        
        /* rules are:
        0 (scalar_exp)
        1 addition
        2 subtraction
        3 multiplication
        4 division
        5 positive [scalar_exp]
        6 negative [scalar_exp] */
        
        if (v->rule_Flag == 0) v->se1->accept(*this);
        else if (v->rule_Flag == 1) {
            v->se1->accept(*this);
            printTabs(NONE);
            cout << "\t<ADDITION>" << endl;
            v->se2->accept(*this);
        }
        else if (v->rule_Flag == 2) {
            v->se1->accept(*this);
            printTabs(NONE);
            cout << "\t<SUBTRACTION>" << endl;
            v->se2->accept(*this);
        }
        else if (v->rule_Flag == 3) {
            v->se1->accept(*this);
            printTabs(NONE);
            cout << "\t<MULTIPLICATION>" << endl;
            v->se2->accept(*this);
        }
        else if (v->rule_Flag == 4) {
            v->se1->accept(*this);
            printTabs(NONE);
            cout << "\t<DIVISION>" << endl;
            v->se2->accept(*this);
        }
        else if (v->rule_Flag == 5) {
            printTabs(NONE);
            cout << "\t<POSITIVE>" << endl;
            v->se1->accept(*this);
        }
        else if (v->rule_Flag == 6) {
            printTabs(NONE);
            cout << "\t<NEGATIVE>" << endl;
            v->se1->accept(*this);
        }
        else {
            if (v->a) v->a->accept(*this);
            if (v->cr) v->cr->accept(*this);
            if (v->fr) v->fr->accept(*this);
        }

        printTabs(DECR);
        cout << "</ScalarExp>" << endl;
    }

    void visit(class SearchCondition *v) {
        printTabs(INCR);
        cout << "<SearchCondition>" << endl;
        /* rules are:
        -1 Predicate
        0 OR
        1 AND
        2 NOT
        3 (search_condition) */
        
        if (v->rule_Flag == 0) {
            v->sc1->accept(*this);
            cout << "<OR>" << endl;
            v->sc2->accept(*this);
        }
        else if (v->rule_Flag == 1) {
            v->sc1->accept(*this);
            cout << "<AND>" << endl;
            v->sc2->accept(*this);
        }
        else if (v->rule_Flag == 2) {
            cout << "<NOT>" << endl;
            v->sc1->accept(*this);
        }
        else if (v->rule_Flag == 3) 
            v->sc1->accept(*this);
        else v->p->accept(*this);

        printTabs(DECR);
        cout << "</SearchCondition>" << endl;
    }
    
    void visit(class LikePredicate *v) {
        printTabs(INCR);
        cout << "<LikePredicate>" << endl;

        v->se->accept(*this);
        if (v->rule_Flag == 1) cout << "<NOT>" << endl;
        cout << "<LIKE>" << endl;
        v->a->accept(*this);
        if (v->oe) v->oe->accept(*this);

        printTabs(DECR);
        cout << "</LikePredicate>" << endl;
    }

    void visit(class ComparisonPredicate *v) {
        printTabs(INCR);
        cout << "<ComparisonPredicate>" << endl;

        v->se1->accept(*this);
        printTabs(NONE);
        cout << "\t<COMPARISON>" << endl;
        if (v->se2) v->se2->accept(*this);
        if (v->s) v->s->accept(*this);

        printTabs(DECR);
        cout << "</ComparisonPredicate>" << endl;
    }

    void visit(class BetweenPredicate *v) {
        printTabs(INCR);
        cout << "<BetweenPredicate>" << endl;

        v->se1->accept(*this);
        if (v->rule_Flag == 1) cout << "<NOT>" << endl;
        cout << "<BETWEEN>" << endl;
        v->se2->accept(*this);
        cout << "<AND>" << endl;
        v->se3->accept(*this);

        printTabs(DECR);
        cout << "</BetweenPredicate>" << endl;
    }

    void visit(class Predicate *v) {
        printTabs(INCR);
        cout << "<Predicate>" << endl;

        if (v->cp) v->cp->accept(*this);
        if (v->bp) v->bp->accept(*this);
        if (v->lp) v->lp->accept(*this);

        printTabs(DECR);
        cout << "</Predicate>" << endl;
    }

    void visit(class FunctionRef *v) {
        printTabs(INCR);
        cout << "<FunctionRef>" << endl;

        if (v->rule_Flag == -1) {
            if (v->cr) {
                v->am->accept(*this);
                cout << "<DISTINCT>" << endl;
                v->cr->accept(*this);
            }
            else {
                v->am->accept(*this);
                cout << "<*>" << endl;
            }
        }
        else if (v->rule_Flag == 0) {
            v->am->accept(*this);
            cout << "<ALL>" << endl;
            v->se->accept(*this);
        }
        else if (v->rule_Flag == 1) {
            v->am->accept(*this);
            v->se->accept(*this);
        }

        printTabs(DECR);
        cout << "</FunctionRef>" << endl;
    }
    
    void visit(class OptAllDistinct *v) {
        printTabs(INCR);
        cout << "<OptAllDistinct ddlCmd='" << v->ddlCmd << "'>" << endl;

        printTabs(DECR);
        cout << "</OptAllDistinct>" << endl;
    }
    
    void visit(class SelectStatement *v) {
        printTabs(INCR);
        cout << "<SelectStatement>" << endl;

        v->OAD->accept(*this);
        v->sel->accept(*this);
        v->tblExp->accept(*this);

        printTabs(DECR);
        cout << "</SelectStatement>" << endl;
    }
    
    void visit(class ManipulativeStatement *v) {
        printTabs(INCR);
        cout << "<ManipulativeStatement>" << endl;

        if (v->selSta) v->selSta->accept(*this);
        if (v->USP) v->USP->accept(*this);
        if (v->USS) v->USS->accept(*this);
        if (v->inSta) v->inSta->accept(*this); 

        printTabs(DECR);
        cout << "</ManipulativeStatement>" << endl;
    }

    void visit(class Schema *v) {
        printTabs(INCR);
        cout << "<Schema>" << endl;

        v->btd->accept(*this);

        printTabs(DECR);
        cout << "</Schema>" << endl;
    }
    
    void visit(class SQL *v) {
        printTabs(INCR);
        cout << "<SQL>" << endl;
        
        if (v->sch) v->sch->accept(*this);
        if (v->manSta) v->manSta->accept(*this);

        printTabs(DECR);
        cout << "</SQL>" << endl;
    }

    void visit(class SQLList *v) {
        printTabs(INCR);
        cout << "<SQLList>" << endl;
        

        if (v->sqlList) { v->sqlList->accept(*this);  }
        v->sql->accept(*this);

        printTabs(DECR);
        cout << "</SQLList>" << endl;
    }


    void visit(class Program *v) {
        printTabs(NONE);
        cout << "<Program>" << endl;

        v->sqlList->accept(*this);

        printTabs(DECR);
        cout << "</Program>" << endl;
    }

    void visit(class DataType *v) {
        printTabs(NONE);

        if (v->size1 != 0) 
            cout << "<DataType flag=" << v->dataType_Flag << "size1" << v->size1 << "/>" << endl;
        
        else 
            cout << "<DataType flag=" << v->dataType_Flag << "/>" << endl;
       
        //printTabs(DECR);
    }

    void visit(class Literal *v) {
        printTabs(INCR);

        if (v->name1 != "") cout << "<Literal string='" << v->name1 << "'/>" << endl;
        if (v->int1 != 0) cout << "<Literal int = " << v->int1 << "/>" << endl;

        printTabs(DECR);
        cout << "</Literal>" << endl;
    }

    void visit(class ColumnRef *v) {
        printTabs(INCR);

        if (v->args == 1)
            cout << "<ColumnRef name='" << v->name1 << "'/>" << endl;
        
        else if (v->args == 2)
            cout << "<ColumnRef name='" << v->name1 << "." << v->name2 << "'/>" << endl;

        else if (v->args == 3)
            cout << "<ColumnRef name='" << v->name1 << "." << v->name2 << "." << v->name3 << "'/>" << endl;

        printTabs(DECR);
        cout << "</ColumnRef>" << endl;
    }

    void visit(class ColumnRefCommalist *v) {

        if (v->crc) v->crc->accept(*this);
        v->cr->accept(*this);

    }

    void visit(class Cursor *v) {
        printTabs(INCR);

        cout << "<Cursor name='" << v->name1 << "'/>" << endl;
        
        printTabs(DECR);
        cout << "</Cursor>" << endl;
    }

    void visit(class Column *v) {
        printTabs(INCR);

        cout << "<Column name='" << v->name1 << "'/>" << endl;
        
        printTabs(DECR);
        cout << "</Column>" << endl;
    }

    void visit(class OptAscDesc *v) {
        printTabs(INCR);

        cout << "<OptAscDesc function='" << v->ascDesc << "'/>" << endl;
        
        printTabs(DECR);
        cout << "</OptAscDesc>" << endl;
    }

    void visit(class Ammsc *v) {
        printTabs(INCR);

        cout << "<Ammsc function='" << v->funcName << "'/>" << endl;
        
        printTabs(DECR);
        cout << "</Ammsc>" << endl;
    }

    void visit(class Table *v) {
        printTabs(INCR);

        //cout << "name1 is |" << v->name1 << "| and name 2 is |" << v->name2 << "|" << endl;
        if (v->name2 != "")
            cout << "<Table name1='" << v->name1 << "' name2='" << v->name2 << "' />" << endl;
        else
            cout << "<Table name='" << v->name1 << "'/>" << endl;
        
        printTabs(DECR);
        cout << "</Table>" << endl;
    }

private:
    static int tabLevel_;   /**< Keeps track of number of tabs to print out on a line. */
};

// Definition of static memble for SimplePrinterVisitor
int XMLTranslator::tabLevel_ = 0;

#endif // AST_SIMPLE_PRINTER_VISITOR_H
