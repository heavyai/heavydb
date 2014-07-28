%name Parser
%define LSP_NEEDED
%define MEMBERS                 \
    virtual ~Parser()   {} \
    int parse(const string & inputStr, ASTNode *& parseRoot, string &lastParsed) { istringstream ss(inputStr); lexer.switch_streams(&ss,0);  yyparse(parseRoot); lastParsed = lexer.YYText(); return yynerrs; } \
    private:                   \
       yyFlexLexer lexer;
%define LEX_BODY {return lexer.yylex();}
%define ERROR_BODY {cerr << "Syntax error on line " << lexer.lineno() << ". Last word parsed:" << lexer.YYText() << endl;}

%header{
#include <iostream>
#include <fstream>
#include <FlexLexer.h>
#include <cstdlib>
#include <string>
#include <sstream>

// AST nodes
#include "ast/ASTNode.h"
#include "ast/AbstractScalarExpr.h"
#include "ast/Program.h"
#include "ast/SQLList.h"
#include "ast/SQL.h"
#include "ast/Schema.h"
#include "ast/BaseTableDef.h"
#include "ast/Table.h"
#include "ast/ColumnDef.h"
#include "ast/ColumnCommalist.h"
#include "ast/TableConstraintDef.h"
#include "ast/BaseTableElementCommalist.h"
#include "ast/BaseTableElement.h"
#include "ast/ColumnDefOpt.h"
#include "ast/ColumnDefOptList.h"
#include "ast/Literal.h"
#include "ast/DataType.h"
#include "ast/Column.h"

#include "ast/ManipulativeStatement.h"
#include "ast/SelectStatement.h"
#include "ast/Selection.h"
#include "ast/OptAllDistinct.h"
#include "ast/TableExp.h"
#include "ast/FromClause.h"
#include "ast/TableRefCommalist.h"
#include "ast/TableRef.h"

#include "ast/InsertStatement.h"
#include "ast/OptColumnCommalist.h"
#include "ast/ValuesOrQuerySpec.h"
#include "ast/QuerySpec.h"
#include "ast/InsertAtomCommalist.h"
#include "ast/InsertAtom.h"
#include "ast/Atom.h"

#include "ast/SearchCondition.h"
#include "ast/ScalarExpCommalist.h"
#include "ast/ScalarExp.h"
#include "ast/FunctionRef.h"
#include "ast/Ammsc.h"
#include "ast/Predicate.h"
#include "ast/ComparisonPredicate.h"
#include "ast/BetweenPredicate.h"
#include "ast/LikePredicate.h"
#include "ast/OptEscape.h"
#include "ast/ColumnRef.h"

#include "ast/ColumnRefCommalist.h"
#include "ast/OptWhereClause.h"
#include "ast/OptGroupByClause.h"
#include "ast/OptHavingClause.h"
#include "ast/OptLimitClause.h"
#include "ast/OptAscDesc.h"
#include "ast/OrderingSpecCommalist.h"
#include "ast/OrderingSpec.h"
#include "ast/OptOrderByClause.h"

#include "ast/UpdateStatementSearched.h"
#include "ast/UpdateStatementPositioned.h"
#include "ast/AssignmentCommalist.h"
#include "ast/Assignment.h"
#include "ast/Cursor.h"

#include "ast/TestForNull.h"
#include "ast/InPredicate.h"
#include "ast/ExistenceTest.h"
#include "ast/AllOrAnyPredicate.h"
#include "ast/AnyAllSome.h"
#include "ast/AtomCommalist.h"
#include "ast/Subquery.h"
#include "ast/GroupByList.h"

using namespace std;
using namespace SQL_Namespace;

// define stack element type to be a 
// pointer to an AST node
#define YY_Parser_STYPE ASTNode*
#define YY_Parser_PARSE_PARAM ASTNode*& parseRoot

extern ASTNode* parse_root;

// Variables declared in scanner.l
extern std::vector<std::string> strData;
extern std::vector<double> realData;
extern std::vector<long int> intData;

extern int mycolno;
extern int mylineno;

%}
%left OR
%left AND
%left NOT
%left <sSubtok> COMPARISON /* = <> < > <= >= */
%left '+' '-'
%left '*' '/' '.'
%nonassoc UMINUS

%token AS
%token DROP
%token NAME
%token TABLE
%token CREATE

%token INTNUM STRING APPROXNUM
%token UNKNOWN

%token ALL BETWEEN BY DISTINCT FROM GROUP HAVING
%token SELECT COMPARISON
%token USER WHERE WITH 
%token EMPTY SELALL DOT 

%token UPDATE SET CURRENT OF NULLX ASSIGN

%token INSERT INTO VALUES

%token NOT UNIQUE PRIMARY FOREIGN KEY CHECK REFERENCES DEFAULT
%token DATATYPE
%token DECIMAL SMALLINT NUMERIC CHARACTER INTEGER REAL FLOAT DOUBLE PRECISION VARCHAR

%token AVG MAX MIN SUM COUNT

%token ALIAS INTORDER COLORDER ORDER ASC DESC
%token LIMIT OFFSET

%token DOTNAME ESCAPE LIKE GROUP BY
%token IS IN ANY SOME EXISTS FOR FSUBSTRING

%start program

%%

program:
    sql_list						{ $$ = new Program((SQLList*)$1); parseRoot = $$; }
|	      							{ $$ = 0; parseRoot = $$; }
;

sql_list:
    sql ';'						    { $$ = new SQLList((SQL*)$1); }
|   sql_list sql ';'				{ $$ = new SQLList((SQLList*)$1, (SQL*)$2); }
;

sql:
    schema 						   { $$ = new SQL((Schema*)$1); }
;

opt_column_commalist:
        /* empty */                { $$ = new OptColumnCommalist(NULL); }
    |   '(' column_commalist ')'   { $$ = new OptColumnCommalist((ColumnCommalist*)$2); }
    ;

schema:
    base_table_def			   	   { $$ = new Schema((BaseTableDef*)$1); }
;

base_table_def:
    CREATE TABLE table '(' base_table_element_commalist ')'		{ $$ = new BaseTableDef("CREATE", (Table*)$3, (BaseTableElementCommalist*)$5); }
| DROP TABLE table 												{ $$ = new BaseTableDef("DROP", (Table*)$3); }
;

base_table_element_commalist:
    base_table_element                                      { $$ = new BaseTableElementCommalist( (BaseTableElement*)$1); }
| base_table_element_commalist ',' base_table_element     { $$ = new BaseTableElementCommalist( (BaseTableElementCommalist*)$1, (BaseTableElement*)$3); }
;

base_table_element:
    column_def              { $$ = new BaseTableElement( (ColumnDef*)$1); }
|   table_constraint_def    { $$ = new BaseTableElement( (TableConstraintDef*)$1); }
;

column_def:
    column data_type column_def_opt_list        { $$ = new ColumnDef( (Column*)$1, (DataType*)$2, (ColumnDefOptList*)$3); }
;

column_def_opt_list:
/* empty */                                 { $$ = NULL; }
| column_def_opt_list column_def_opt        { $$ = new ColumnDefOptList( (ColumnDefOptList*)$1, (ColumnDefOpt*)$2); }
;

column_def_opt:
													// ColumnDefOpt(int rule_Flags, Literal* lit, SearchCondition* srchCon, Table* tbl, ColumnCommalist* colComList);
        NOT NULLX                                   { $$ = new ColumnDefOpt(0); }
    |   NOT NULLX UNIQUE                            { $$ = new ColumnDefOpt(1); }
    |   NOT NULLX PRIMARY KEY                       { $$ = new ColumnDefOpt(2); }
    |   DEFAULT literal                             { $$ = new ColumnDefOpt(3, (Literal*)$2); }
    |   DEFAULT NULLX                               { $$ = new ColumnDefOpt(4); }
    |   DEFAULT USER                                { $$ = new ColumnDefOpt(5); }
    |   CHECK '(' search_condition ')'              { $$ = new ColumnDefOpt(6, (SearchCondition*)$3); }
    |   REFERENCES table                            { $$ = new ColumnDefOpt(7, (Table*)$2); }
    |   REFERENCES table '(' column_commalist ')'   { $$ = new ColumnDefOpt(8, (Table*)$2, (ColumnCommalist*)$4); }
    ;

table_constraint_def:
    UNIQUE '(' column_commalist ')'                                                   { $$ = new TableConstraintDef(0, (ColumnCommalist*)$3); }
|   PRIMARY KEY '(' column_commalist ')'                                              { $$ = new TableConstraintDef(1, (ColumnCommalist*)$4); }
|   FOREIGN KEY '(' column_commalist ')' REFERENCES table                             { $$ = new TableConstraintDef(2, (ColumnCommalist*)$4, (Table*)$7); }
|   FOREIGN KEY '(' column_commalist ')' REFERENCES table '(' column_commalist ')'    { $$ = new TableConstraintDef(2, (ColumnCommalist*)$4, (Table*)$7, (ColumnCommalist*)$9); }
|   CHECK '(' search_condition ')'                                                  {$$ = new TableConstraintDef(3, (SearchCondition*)$3); }
;

column_commalist:
        column                                      { $$ = new ColumnCommalist((Column*)$1); }
    |   column_commalist ',' column                 { $$ = new ColumnCommalist((ColumnCommalist*)$1, (Column*)$3); }
    ;

opt_order_by_clause:
        /* empty */                             { $$ = NULL; }
    |   ORDER BY ordering_spec_commalist        { $$ = new OptOrderByClause((OrderingSpecCommalist*)$3); }
    ;
    
ordering_spec_commalist:
        ordering_spec                                   { $$ = new OrderingSpecCommalist((OrderingSpec*)$1); }
    |   ordering_spec_commalist ',' ordering_spec       { $$ = new OrderingSpecCommalist((OrderingSpecCommalist*)$1, (OrderingSpec*)$3); }
    ;

ordering_spec:
        INTNUM opt_asc_desc             { $$ = new OrderingSpec(realData[0], (OptAscDesc*)$2); }
    |   column_ref opt_asc_desc         { $$ = new OrderingSpec((ColumnRef*)$1, (OptAscDesc*)$2); }
    ;

opt_asc_desc:
        /* empty */                     { $$ = new OptAscDesc(""); }
    |   ASC                             { $$ = new OptAscDesc("ASC"); }
    |   DESC                            { $$ = new OptAscDesc("DESC"); }
    ;

/*************/
/* this starts the execution of classic manipulative statements. */

sql:
    manipulative_statement      { $$ = new SQL((ManipulativeStatement*)$1); }
;

manipulative_statement:
    select_statement                    { $$ = new ManipulativeStatement((SelectStatement*)$1); } 
    |   insert_statement                    { $$ = new ManipulativeStatement((InsertStatement*)$1); }
    |   update_statement_positioned         { $$ = new ManipulativeStatement((UpdateStatementPositioned*)$1); }
    |   update_statement_searched           { $$ = new ManipulativeStatement((UpdateStatementSearched*)$1); }
  /*
    
    |   commit_statement
    |   delete_statement_positioned
    |   delete_statement_searched
    |   fetch_statement
    |   insert_statement
    |   open_statement
    |   rollback_statement 
    |   select_statement 
    |   update_statement_positioned
    |   update_statement_searched */
    ;

insert_statement:
    INSERT INTO table opt_column_commalist values_or_query_spec {
        $$ = new InsertStatement((Table*)$3, (OptColumnCommalist*)$4, (ValuesOrQuerySpec*)$5);
    }
;

values_or_query_spec:
        VALUES '(' insert_atom_commalist ')'        { $$ = new ValuesOrQuerySpec((InsertAtomCommalist*)$3); }
    |   query_spec                                  { $$ = new ValuesOrQuerySpec((QuerySpec*)$1); }         
    ;

insert_atom_commalist:
        insert_atom                                 { $$ = new InsertAtomCommalist((InsertAtom*)$1); } 
    |   insert_atom_commalist ',' insert_atom       { $$ = new InsertAtomCommalist((InsertAtomCommalist*)$1, (InsertAtom*)$3); } 
    ;

insert_atom:
        atom                                        { $$ = new InsertAtom((Atom*)$1); }
    |   NULLX                                       { $$ = new InsertAtom(NULL); }
    ;

select_statement:
        SELECT opt_all_distinct selection
      /*   INTO target_commalist */ 
         table_exp                                  { $$ = new SelectStatement((OptAllDistinct*)$2, (Selection*)$3, (TableExp*)$4); }   
    ;

opt_all_distinct:
    ALL                                     { $$ = new OptAllDistinct("ALL"); }
    |   DISTINCT                            { $$ = new OptAllDistinct("DISTINCT"); }
    | /* empty  */                          { $$ = new OptAllDistinct(""); }
    ;

update_statement_positioned:
        UPDATE table SET assignment_commalist       
        WHERE CURRENT OF cursor                 { $$ = new UpdateStatementPositioned((Table*)$2, (AssignmentCommalist*)$4, (Cursor*)$8); }
    ;

assignment_commalist:
         /* empty */                            { $$ = NULL; }
    |   assignment                              { $$ = new AssignmentCommalist((Assignment*)$1); }
    |   assignment_commalist ',' assignment     { $$ = new AssignmentCommalist((AssignmentCommalist*)$1, (Assignment*)$3); }
    ;

assignment:
        column COMPARISON scalar_exp               { $$ = new Assignment((Column*)$1, (ScalarExp*)$3); }
    |   column COMPARISON NULLX                    { $$ = new Assignment((Column*)$1, NULL); }
    ;    

update_statement_searched:
        UPDATE table SET assignment_commalist opt_where_clause      { $$ = new UpdateStatementSearched((Table*)$2, (AssignmentCommalist*)$4, (OptWhereClause*)$5); }
    ;

query_spec:
        SELECT opt_all_distinct selection table_exp     { $$ = new QuerySpec((OptAllDistinct*)$2, (Selection*)$3, (TableExp*)$4); }
    ;

selection:
    scalar_exp_commalist            { $$ = new Selection((ScalarExpCommalist*)$1); }
| '*'                               { $$ = new Selection("*"); }
;

table_exp:
    from_clause            
    opt_where_clause 
    opt_group_by_clause 
    opt_having_clause
    opt_order_by_clause
    opt_limit_clause                            { $$ = new TableExp((FromClause*)$1, (OptWhereClause*)$2, (OptGroupByClause*)$3, (OptHavingClause*)$4,
                                                    (OptOrderByClause*)$5, (OptLimitClause*)$6); }
;

from_clause
: FROM table_ref_commalist              { $$ = new FromClause((TableRefCommalist*)$2); }
| FROM '(' select_statement ')'         { $$ = new FromClause((SelectStatement*)$3); }
;

table_ref_commalist:
        table_ref                               { $$ = new TableRefCommalist((TableRef*)$1); }
    |   table_ref_commalist ',' table_ref       { $$ = new TableRefCommalist((TableRefCommalist*)$1, (TableRef*)$3); }
    ;
    
table_ref:
    table                                       { $$ = new TableRef((Table *)$1); }
    /* | table rangevariable */
    ;

opt_where_clause:
    WHERE search_condition                      { $$ = new OptWhereClause((SearchCondition*)$2); }
    | /* empty */                               { $$ = NULL; }
    ;

opt_group_by_clause:
        /* empty */                             { $$ = NULL; }
    //|   GROUP BY column_ref_commalist           { $$ = new OptGroupByClause((ColumnRefCommalist*)$3); }
    |   GROUP BY group_by_list 
    /*opt_with_rollup */            { $$ = new OptGroupByClause((GroupByList*)$3); }
    ;

group_by_list:
    scalar_exp opt_asc_desc                         { $$ = new GroupByList((ScalarExp*)$1, (OptAscDesc*)$2); }
    | group_by_list ',' scalar_exp opt_asc_desc     { $$ = new GroupByList((GroupByList*)$1, (ScalarExp*)$3, (OptAscDesc*)$4); }
;
/*
column_ref_commalist:   
        column_ref                              { $$ = new ColumnRefCommalist((ColumnRef*)$1); }
    |   column_ref_commalist ',' column_ref     { $$ = new ColumnRefCommalist((ColumnRefCommalist*)$1, (ColumnRef*)$3); }
    ;
*/
opt_having_clause:
    HAVING search_condition                 { $$ = new OptHavingClause((SearchCondition*)$2); }
    | /* empty */                           { $$ = NULL; }
    ;

opt_limit_clause:
    /* empty */                               { $$ = NULL; }
    | LIMIT INTNUM                            { $$ = new OptLimitClause(realData[0]); }
//    | LIMIT INTNUM ',' INTNUM                 { $$ = new OptLimitClause(0, realData[0], realData[1]); }
    | LIMIT INTNUM OFFSET INTNUM              { $$ = new OptLimitClause(1, realData[0], realData[1]); }
    /* search conditions */

search_condition: 
      search_condition OR search_condition      { $$ = new SearchCondition(0, (SearchCondition*)$1, (SearchCondition*)$3); }
    |   search_condition AND search_condition   { $$ = new SearchCondition(1, (SearchCondition*)$1, (SearchCondition*)$3); }
    |   NOT search_condition                    { $$ = new SearchCondition(2, (SearchCondition*)$2); }
    |   '(' search_condition ')'                { $$ = new SearchCondition(3, (SearchCondition*)$2); }
    |   predicate                               { $$ = new SearchCondition((Predicate*)$1); }
    ;

predicate:
        comparison_predicate                    { $$ = new Predicate((ComparisonPredicate*)$1); }
    |   between_predicate                       { $$ = new Predicate((BetweenPredicate*)$1); }
    |   like_predicate                          { $$ = new Predicate((LikePredicate*)$1); }
    |   test_for_null                           { $$ = new Predicate((TestForNull*)$1); }
    |   in_predicate                            { $$ = new Predicate((InPredicate*)$1); }
    |   all_or_any_predicate                    { $$ = new Predicate((AllOrAnyPredicate*)$1); }
    |   existence_test                          { $$ = new Predicate((ExistenceTest*)$1); }
    ;

comparison_predicate:
                                                                // strData[5] is hardcoded to contain comparison value
        scalar_exp COMPARISON scalar_exp                        { $$ = new ComparisonPredicate(strData[5], (ScalarExp*)$1, (ScalarExp*)$3); }
    |   scalar_exp COMPARISON subquery                          { $$ = new ComparisonPredicate(strData[5], (ScalarExp*)$1, (Subquery*)$3); }
    ;

between_predicate:
        scalar_exp NOT BETWEEN scalar_exp AND scalar_exp        { $$ = new BetweenPredicate(2, (ScalarExp*)$1, (ScalarExp*)$4, (ScalarExp*)$6); }
    |   scalar_exp BETWEEN scalar_exp AND scalar_exp            { $$ = new BetweenPredicate(1, (ScalarExp*)$1, (ScalarExp*)$3, (ScalarExp*)$5); }
    ;

like_predicate:
        scalar_exp NOT LIKE atom opt_escape                     { $$ = new LikePredicate(2, (ScalarExp*)$1, (Atom*)$4, (OptEscape*)$5); }
    |   scalar_exp LIKE atom opt_escape                         { $$ = new LikePredicate(1, (ScalarExp*)$1, (Atom*)$3, (OptEscape*)$4); }
    ;

opt_escape:
        /* empty */                                             { $$ = NULL; }
    |   ESCAPE atom                                             { $$ = new OptEscape((Atom*)$2); }
    ;

test_for_null:
        column_ref IS NOT NULLX                                 { $$ = new TestForNull(1, (ColumnRef*)$1); }
    |   column_ref IS NULLX                                     { $$ = new TestForNull(0, (ColumnRef*)$1); }
    ;

in_predicate:
        scalar_exp NOT IN '(' subquery ')'                      { $$ = new InPredicate(1, (ScalarExp*)$1, (Subquery*)$5); }
    |   scalar_exp IN '(' subquery ')'                          { $$ = new InPredicate(0, (ScalarExp*)$1, (Subquery*)$4); }
    |   scalar_exp NOT IN '(' atom_commalist ')'                { $$ = new InPredicate(1, (ScalarExp*)$1, (AtomCommalist*)$5); }
    |   scalar_exp IN '(' atom_commalist ')'                    { $$ = new InPredicate(0, (ScalarExp*)$1, (AtomCommalist*)$4); }
    ;

atom_commalist:
        atom                                                    { $$ = new AtomCommalist((Atom*)$1); }
    |   atom_commalist ',' atom                                 { $$ = new AtomCommalist((AtomCommalist*)$1, (Atom*)$3); }
    ;

all_or_any_predicate:
        scalar_exp COMPARISON any_all_some subquery             { $$ = new AllOrAnyPredicate((ScalarExp*)$1, (AnyAllSome*)$3, (Subquery*)$4); }
    ;
            
any_all_some:
        ANY                                                     { $$ = new AnyAllSome("ANY"); }
    |   ALL                                                     { $$ = new AnyAllSome("ALL"); }
    |   SOME                                                    { $$ = new AnyAllSome("SOME"); }
    ;

existence_test:
        EXISTS subquery                                         { $$ = new ExistenceTest((Subquery*)$2); }
    ;

subquery:
        '(' SELECT opt_all_distinct selection table_exp ')'     { $$ = new Subquery((OptAllDistinct*)$3, (Selection*)$4, (TableExp*)$5); }
    ;

/* scalar stuff */

scalar_exp_commalist:       
        scalar_exp                              { $$ = new ScalarExpCommalist((ScalarExp*)$1); }
    |   scalar_exp_commalist ',' scalar_exp     { $$ = new ScalarExpCommalist((ScalarExpCommalist*)$1, (ScalarExp*)$3); }
    ;

scalar_exp:
    scalar_exp '+' scalar_exp               { $$ = new ScalarExp(1, (ScalarExp*)$1, (ScalarExp*)$3); }
    | scalar_exp '-' scalar_exp             { $$ = new ScalarExp(2, (ScalarExp*)$1, (ScalarExp*)$3); }
    | scalar_exp '*' scalar_exp             { $$ = new ScalarExp(3, (ScalarExp*)$1, (ScalarExp*)$3); }
    | scalar_exp '/' scalar_exp             { $$ = new ScalarExp(4, (ScalarExp*)$1, (ScalarExp*)$3); }
    |   '+' scalar_exp %prec UMINUS         { $$ = new ScalarExp(5, (ScalarExp*)$2);  }
    |   '-' scalar_exp %prec UMINUS         { $$ = new ScalarExp(6, (ScalarExp*)$2); }
    |   atom                                { $$ = new ScalarExp((Atom*)$1); }
    |   column_ref                          { $$ = new ScalarExp((ColumnRef*)$1); }
    |   function_ref                        { $$ = new ScalarExp((FunctionRef*)$1); }
    |   '(' scalar_exp ')'                  { $$ = new ScalarExp(0, (ScalarExp*)$2); }
    ;

atom:
    literal { $$ = new Atom((Literal*)$1);  }
;

function_ref:
        ammsc '(' '*' ')'                   { $$ = new FunctionRef((Ammsc*)$1);}
    |   ammsc '(' DISTINCT column_ref ')'   { $$ = new FunctionRef((Ammsc*)$1, (ColumnRef*)$4); }
    |   ammsc '(' ALL scalar_exp ')'        { $$ = new FunctionRef(0, (Ammsc*)$1, (ScalarExp*)$4); }
    |   ammsc '(' scalar_exp ')'            { $$ = new FunctionRef(1, (Ammsc*)$1, (ScalarExp*)$3); }
    |   FSUBSTRING '(' scalar_exp_commalist ')'                         { $$ = new FunctionRef("substr", (ScalarExpCommalist*)$3); }
    |   FSUBSTRING '(' scalar_exp FROM scalar_exp ')'                   { $$ = new FunctionRef("substr", (ScalarExp*)$3, (ScalarExp*)$5); }
    |   FSUBSTRING '(' scalar_exp FROM scalar_exp FOR scalar_exp ')'    { $$ = new FunctionRef("substr", (ScalarExp*)$4, (ScalarExp*)$5, (ScalarExp*)$7); }
    ;

literal
: STRING /* should be: STRING */ {
        assert(realData.size() != 0);
        $$ = new Literal(strData.back());
        strData.pop_back();
        ((Literal*)$$)->setLineno(mylineno);
        ((Literal*)$$)->setColno(mycolno - lexer.YYLeng());
    }
| INTNUM {
        assert(intData.size() != 0);
        $$ = new Literal(intData.back());
        intData.pop_back();
    }
| APPROXNUM {
        assert(realData.size() != 0);
        $$ = new Literal(realData.back());
        realData.pop_back();
    }
;

table:
  NAME {
    // printf("table: [%d] strData.size()=%d \"%s\"\n", __LINE__, strData.size(), strData[0].c_str()); 
    assert(strData.size() != 0); 
    $$ = new Table(strData.back());
    strData.pop_back();
}
| NAME '.' NAME     { $$ = new Table(0, strData[0], strData[1]); }
| NAME AS NAME      { $$ = new Table(1, strData[0], strData[1]); }
;


/* data types */
data_type
: CHARACTER                           { $$ = new DataType(0); }
| CHARACTER '(' INTNUM ')'            { $$ = new DataType(0, realData[0]); }
| VARCHAR                             { $$ = new DataType(1); }
| VARCHAR '(' INTNUM ')'              { $$ = new DataType(1, realData[0]); }
| NUMERIC                             { $$ = new DataType(2); }
| NUMERIC '(' INTNUM ')'              { $$ = new DataType(2, realData[0]); }
| NUMERIC '(' INTNUM ',' INTNUM ')'   { $$ = new DataType(2, realData[0], realData[1]); }
| DECIMAL                             { $$ = new DataType(3); }
| DECIMAL '(' INTNUM ')'              { $$ = new DataType(3, realData[0]); }
| DECIMAL '(' INTNUM ',' INTNUM ')'   { $$ = new DataType(3, realData[0], realData[1]); }
| INTEGER                             { $$ = new DataType(4); }
| SMALLINT                            { $$ = new DataType(5); }
| FLOAT                               { $$ = new DataType(6); }
| FLOAT '(' INTNUM ')'                { $$ = new DataType(6, realData[0]); }
| REAL                                { $$ = new DataType(7); }
| DOUBLE PRECISION                    { $$ = new DataType(8); }
;

column_ref:
        NAME                    { printf("[%d]\n", __LINE__); $$ = new ColumnRef(strData[0]); }
    |   NAME '.' NAME           { $$ = new ColumnRef(0, strData[0], strData[1]); }
    |   NAME '.' NAME '.' NAME  { $$ = new ColumnRef(strData[0], strData[1], strData[2]);}
    |   NAME AS NAME            { $$ = new ColumnRef(1, strData[0], strData[1]); } 
;

column:
    NAME {
        assert(strData.size() > 0); 
        $$ = new Column(strData.back()); 
        strData.pop_back();
    }
;

cursor
: NAME            { $$ = new Cursor(strData[0]); }
;

ammsc: 
    AVG             { $$ = new Ammsc("AVG"); }
    | MIN           { $$ = new Ammsc("MIN"); }
    | MAX           { $$ = new Ammsc("MAX"); }
    | SUM           { $$ = new Ammsc("SUM"); }
    | COUNT         { $$ = new Ammsc("COUNT"); }
;

%%
