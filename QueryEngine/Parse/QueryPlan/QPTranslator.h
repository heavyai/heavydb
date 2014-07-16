#ifndef AST_SIMPLE_TRANSLATOR_VISITOR_H
#define AST_SIMPLE_TRANSLATOR_VISITOR_H

#include "SQL/visitor/Visitor.h"
#include "RA/visitor/Visitor.h"

#include "SQL/ast/ColumnCommalist.h"
#include "SQL/ast/TableConstraintDef.h"
#include "SQL/ast/BaseTableDef.h"
#include "SQL/ast/BaseTableElementCommalist.h"
#include "SQL/ast/BaseTableElement.h"
#include "SQL/ast/Program.h"
#include "SQL/ast/Table.h"
#include "SQL/ast/Schema.h"
#include "SQL/ast/SQL.h"
#include "SQL/ast/ColumnDef.h"
#include "SQL/ast/ColumnDefOpt.h"
#include "SQL/ast/ColumnDefOptList.h"
#include "SQL/ast/Column.h"
#include "SQL/ast/Literal.h"
#include "SQL/ast/DataType.h"
#include "SQL/ast/SQLList.h"

#include "SQL/ast/ManipulativeStatement.h"
#include "SQL/ast/SelectStatement.h"
#include "SQL/ast/Selection.h"
#include "SQL/ast/OptAllDistinct.h"
#include "SQL/ast/TableExp.h"
#include "SQL/ast/FromClause.h"
#include "SQL/ast/TableRefCommalist.h"
#include "SQL/ast/TableRef.h"

#include "SQL/ast/InsertStatement.h"
#include "SQL/ast/OptColumnCommalist.h"
#include "SQL/ast/ValuesOrQuerySpec.h"
#include "SQL/ast/QuerySpec.h"
#include "SQL/ast/InsertAtomCommalist.h"
#include "SQL/ast/InsertAtom.h"
#include "SQL/ast/Atom.h"

#include "SQL/ast/SearchCondition.h"
#include "SQL/ast/ScalarExpCommalist.h"
#include "SQL/ast/ScalarExp.h"
#include "SQL/ast/FunctionRef.h"
#include "SQL/ast/Ammsc.h"
#include "SQL/ast/Predicate.h"
#include "SQL/ast/ComparisonPredicate.h"
#include "SQL/ast/BetweenPredicate.h"
#include "SQL/ast/LikePredicate.h"
#include "SQL/ast/OptEscape.h"
#include "SQL/ast/ColumnRef.h"

#include "SQL/ast/ColumnRefCommalist.h"
#include "SQL/ast/OptWhereClause.h"
#include "SQL/ast/OptHavingClause.h"
#include "SQL/ast/OptLimitClause.h"
#include "SQL/ast/OptAscDesc.h"
#include "SQL/ast/OrderingSpecCommalist.h"
#include "SQL/ast/OrderingSpec.h"
#include "SQL/ast/OptOrderByClause.h"

#include "SQL/ast/UpdateStatementSearched.h"
#include "SQL/ast/UpdateStatementPositioned.h"
#include "SQL/ast/AssignmentCommalist.h"
#include "SQL/ast/Assignment.h"
#include "SQL/ast/Cursor.h"

#include "SQL/ast/TestForNull.h"
#include "SQL/ast/InPredicate.h"
#include "SQL/ast/ExistenceTest.h"
#include "SQL/ast/AllOrAnyPredicate.h"
#include "SQL/ast/AnyAllSome.h"
#include "SQL/ast/AtomCommalist.h"
#include "SQL/ast/Subquery.h"

//#include "RA/relAlg/RelAlgNode.h"
#include "RA/relAlg/Program.h"
#include "RA/relAlg/RelExprList.h"
#include "RA/relAlg/RelExpr.h"
#include "RA/relAlg/UnaryOp.h"
#include "RA/relAlg/BinaryOp.h"
#include "RA/relAlg/MathExpr.h"
#include "RA/relAlg/SelectOp.h"
#include "RA/relAlg/ProjectOp.h"
#include "RA/relAlg/SortOp.h"
#include "RA/relAlg/ExtendOp.h"
#include "RA/relAlg/GroupByOp.h"
#include "RA/relAlg/RenameOp.h"

#include "RA/relAlg/JoinOp.h"
#include "RA/relAlg/SemijoinOp.h"
#include "RA/relAlg/ProductOp.h"
#include "RA/relAlg/OuterjoinOp.h"
#include "RA/relAlg/AntijoinOp.h"
#include "RA/relAlg/UnionOp.h"
#include "RA/relAlg/AggrExpr.h"
#include "RA/relAlg/AggrList.h"
#include "RA/relAlg/AttrList.h"
#include "RA/relAlg/Attribute.h"
#include "RA/relAlg/Relation.h"
#include "RA/relAlg/Data.h"

#include "RA/relAlg/RA_Predicate.h"
#include "RA/relAlg/Comparison.h"
#include "RA/relAlg/Compared.h"
#include "RA/relAlg/CompOp.h"

#include <iostream>
using std::cout;
using std::endl;

#define TAB_SIZE 2 // number of spaces in a tab

enum tabFlag {INCR, DECR, NONE};
/**
 * @todo brief and detailed descriptions
 */
class QPTranslator : public SQL_Namespace::Visitor {

public:

    // should be a RelExprList, but let's go one at a time for our sanity
    RelExpr* root;

    static void printTabs(tabFlag flag) {
     //   tabLevel_ = 0;
    
        if (flag == INCR)
            tabLevel_++;
        //cout << tabLevel_;
        for (int i = 0; i < tabLevel_; ++i)
            for (int j = 0; j < TAB_SIZE; ++j)
                cout << " ";
        if (flag == DECR)
            tabLevel_--;
    
    }

    RelExpr* getRoot() {
        return root;
    }

    ProductOp* formProductOp(TableRefCommalist *vTRC, TableRef *vTR) {
        // If the table reference commalist does not exist, create a Product Operation of the only table and NULL.
        if (!(vTRC)) 
            return new ProductOp(new RelExpr(new Relation(vTR->tbl)), NULL);
        // IF the table reference commalist exists, create a product operation between the table reference commalist and the table. 
        else 
            return new ProductOp(new RelExpr(formProductOp(vTRC->trc, vTRC->tr)), new RelExpr(new Relation(vTR->tbl)));
    }

    // Begins the recursive examining of the Table-reference list.
    RelExpr* formProductRelation(TableRefCommalist *v) {
        return new RelExpr(formProductOp(v->trc, v->tr));
    }

    MathExpr* handleScalarExp(ScalarExp* se) {
                /* a ScalarExp can take on one of the following values:
        -1 Atom, Column_Ref, or Function_Ref
        0 (scalar_exp)
        1 addition
        2 subtraction
        3 multiplication
        4 division
        5 positive [scalar_exp]
        6 negative [scalar_exp]
         */

        // if rule_Flag is special
        if (se->rule_Flag == -1) {
            // if scalarExp is type atom
            if (se->a) {
                // if the literal in atom contains a string value or a constant
                if (se->a->lit->int1 == 0) return new MathExpr(new Data(se->a->lit->name1));
                else if (se->a->lit->name1 == "") return new MathExpr(new Data(se->a->lit->int1));
            }
            // if scalarExp is type Column_Ref
            else if (se->cr) {
                return new MathExpr(new Attribute(se->cr->name1, se->cr->name2));
            }
            // if scalarExp is type FunctionRef
            else if (se->fr) {
                //check fields of FunctionRef
                if (se->fr->cr) {
                    return new MathExpr(new AggrExpr("DISTINCT", se->fr->am->funcName, new Attribute(se->fr->cr->name1, se->fr->cr->name2)));    
                }
                /* If the function is not DISTINCT we just assume that the function argument is a column ref, even though 
                it can technically nest since the argument of AMMSC (ALL x) or AMMSC (x) is a scalar expression, not a column ref. */
                else return new MathExpr(new AggrExpr("", se->fr->am->funcName, new Attribute(se->fr->se->cr->name1, se->fr->se->cr->name2)));
            }

        // if rule_Flag is between 1 and 4, then MathExpr() takes two arguments
        else if ((1 <= se->rule_Flag) && (se->rule_Flag <= 4))
            return new MathExpr(se->rule_Flag, handleScalarExp(se->se1), handleScalarExp(se->se2));

        // if rule_Flag is 0, 5, or 6, then MathExpr() takes one argument        
        else 
            return new MathExpr(se->rule_Flag, handleScalarExp(se->se1));
        }
    }

    // Break apart the search condition tree and form a predicate tree from the comparisons.
    RA_Predicate* formSelectOp(SearchCondition* sc) {
        /* sc->rule_Flag can take on one of the following values:
        -1 Predicate Node (i.e. contains a comparison, between, like, etc.)
        0 OR
        1 AND
        2 NOT
        3 (search_condition) (i.e. another SearchCondition Node in parens) */
        
        // If rule_Flag is -1, we can create a Relational Algebra predicate from the contents of sc.
        if (sc->rule_Flag == -1) {
            // We are assuming that the only kind of search condition is a comparison predicate, for the mean time.
            if (sc->p->cp) {
                //Also, no subqueries for the meantime: Just a > b.
                Comparison* comp = new Comparison(new Compared(handleScalarExp(sc->p->cp->se1)),
                    new CompOp("Comparison Not Implemented Yet"), new Compared(handleScalarExp(sc->p->cp->se2)));

                return new RA_Predicate(comp);
            }
        }

        //AND, OR
        else if ((sc->rule_Flag == 0) || (sc->rule_Flag == 1)) {
            return new RA_Predicate(sc->rule_Flag, formSelectOp(sc->sc1), formSelectOp(sc->sc2));
        }

        // NOT, (nested)
        else if ((sc->rule_Flag == 2) || (sc->rule_Flag == 3)) {
            return new RA_Predicate(sc->rule_Flag, formSelectOp(sc->sc1));
        }
    }

    // Begins the recursive examining of the predicate tree. The first argument is the relation that is selected upon.
    RelExpr* formSelectRelation(RelExpr* relexArg, TableExp *tblExp) {
        // Check if the where clause is present; if so, break apart the search condition tree and form predicates.
        if (tblExp->owc) 
            return new RelExpr(new SelectOp(relexArg, formSelectOp(tblExp->owc->sc)));
        // otherwise the relational table to be projected is the full list of tables.
        else
            return relexArg;

    }

    // Form an attribute list from the Scalar Expression Commalist
    AttrList* formAttrList(ScalarExpCommalist* sec) {
        // If the ScalarExpCommalist's ScalarExpCommalist child doesn't exist, turn the scalarExp child into an attribute list.
        if (!sec->sec) 
            return new AttrList(new Attribute(sec->se->cr->name1, sec->se->cr->name2));
        
        // If the ScalarExpCommalist's ScalarExpCommalist child does exist, make an AttrList out of that child and make a new one out of the two.
        else if (sec->sec) 
            return new AttrList(formAttrList(sec->sec), new Attribute(sec->se->cr->name1, sec->se->cr->name2));
    }

    //Delve into the scalar_exp_commalist where no man has gone be4    
    RelExpr* formProjectRelation(RelExpr* relexArg, Selection *s) {
        if (s->selectAll != "") return new RelExpr(new ProjectOp(relexArg, s->selectAll)); 
        else return new RelExpr(new ProjectOp(relexArg, formAttrList(s->sec)));
    }

    void visit(class SelectStatement *v) {
      //  printTabs(INCR);
    //    cout << "<SelectStatement>" << endl;

 //       v->OAD->accept(*this);
        v->sel->accept(*this);
        v->tblExp->accept(*this);

        RelExpr* finishedProduct = formProductRelation(v->tblExp->fc->trc);
        RelExpr* voterSelect = formSelectRelation(finishedProduct, v->tblExp);
        RelExpr* finalProject = formProjectRelation(voterSelect, v->sel);

        // for this kind of statement
        root = finalProject;

    //    printTabs(DECR);
//        cout << "</SelectStatement>" << endl;
    }
    
    void visit(class ManipulativeStatement *v) {
     //   printTabs(INCR);
    //    cout << "<ManipulativeStatement>" << endl;

        if (v->selSta) v->selSta->accept(*this);
        if (v->USP) v->USP->accept(*this);
        if (v->USS) v->USS->accept(*this);
        if (v->inSta) v->inSta->accept(*this); 

       // printTabs(DECR);
     //   cout << "</ManipulativeStatement>" << endl;
    }


    void visit(class SQL *v) {
       // printTabs(INCR);
      //  cout << "<SQL>" << endl;
        
        if (v->sch) v->sch->accept(*this);
        if (v->manSta) v->manSta->accept(*this);

     ///   printTabs(DECR);
        //cout << "</SQL>" << endl;
    }

    void visit(class SQLList *v) {
     //   printTabs(INCR);
      //  cout << "<SQLList>" << endl;
        

        if (v->sqlList) { v->sqlList->accept(*this);  }
        v->sql->accept(*this);

     //   printTabs(DECR);
       // cout << "</SQLList>" << endl;
    }


    void visit(class Program *v) {
       // printTabs(NONE);
      //  cout << "<Program>" << endl;

        v->sqlList->accept(*this);

     //   printTabs(DECR);
    //    cout << "</Program>" << endl;
    }



































































































































/*******************************/

    void visit(class BaseTableDef *v) {
        printTabs(INCR);
        cout << "<BaseTableD ddlCmd='" << v->ddlCmd << "'>" << endl;

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
        cout << "SELECT" << endl;

        v->sc->accept(*this);

        printTabs(DECR);
        cout << "</SELECT>" << endl;
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
    //    printTabs(INCR);
    //   cout << "<TableRef>" << endl;

        if (v->tbl) v->tbl->accept(*this);

    //    printTabs(DECR);
    //    cout << "</TableRef>" << endl;
    }
    
    void visit(class TableRefCommalist *v) {
        printTabs(INCR);
        cout << "PRODUCT\n";
        if (v->trc) {
            v->trc->accept(*this);
        }

        v->tr->accept(*this);
        printTabs(DECR);
        cout << "</PRODUCT\n";
    }

    void visit(class FromClause *v) {
      //  printTabs(INCR);
      //  cout << "<FromClause>" << endl;

        if (v->trc) {
          //  printTabs(INCR);
            v->trc->accept(*this);

          //  printTabs(DECR);
            //cout << "</TableRefCommalist>" << endl;
        }
        if (v->ss) v->ss->accept(*this);

    //    printTabs(DECR);
    //    cout << "</FromClause>" << endl;
    }
    
    void visit(TableExp *v) {
   //     printTabs(INCR);
        //cout << "<TableExp>" << endl;


        if (v->owc) v->owc->accept(*this);
        v->fc->accept(*this);
        if (v->ogbc) v->ogbc->accept(*this);
        if (v->ohc) v->ohc->accept(*this);
        if (v->oobc) v->oobc->accept(*this);
        if (v->olc) v->olc->accept(*this);

      //  printTabs(DECR);
        //cout << "</TableExp>" << endl;
    }

    void visit(class Selection *v) {
        printTabs(INCR);
        cout << "<PROJECT>" << endl;

        if(v->sec) {
         //   printTabs(INCR);
         //   cout << "<ScalarExpCommalist>" << endl;
            
            v->sec->accept(*this);

          //  printTabs(DECR);
          //   cout << "</ScalarExpCommalist>" << endl;
        }
        else {
            printTabs(NONE);
            cout << "<*>" << endl;
        }

        printTabs(DECR);
        cout << "</PROJECT>" << endl;
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
    //    printTabs(INCR);
   //     cout << "<ScalarExp>" << endl;
        
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

       // printTabs(DECR);
       // cout << "</ScalarExp>" << endl;
    }

    void visit(class Schema *v) {
        printTabs(INCR);
        cout << "<Schema>" << endl;

        v->btd->accept(*this);

        printTabs(DECR);
        cout << "</Schema>" << endl;
    }
    
    void visit(class SearchCondition *v) {
     //   printTabs(INCR);
     //   cout << "<SearchCondition>" << endl;
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

      //  printTabs(DECR);
     //   cout << "</SearchCondition>" << endl;
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
   //    printTabs(INCR);
      //  cout << "<ComparisonPredicate>" << endl;

        v->se1->accept(*this);
        printTabs(INCR);
        cout << "<COMPARISON>" << endl;
        tabLevel_--;
        if (v->se2) v->se2->accept(*this);
        if (v->s) v->s->accept(*this);

   //     printTabs(DECR);
   //     cout << "</ComparisonPredicate>" << endl;
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
     //   printTabs(INCR);
     //   cout << "<Predicate>" << endl;

        if (v->cp) v->cp->accept(*this);
        if (v->bp) v->bp->accept(*this);
        if (v->lp) v->lp->accept(*this);

       // printTabs(DECR);
      //  cout << "</Predicate>" << endl;
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
int QPTranslator::tabLevel_ = 0;

#endif // AST_SIMPLE_TRANSLATOR_VISITOR_H
