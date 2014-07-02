%name Parser
%define LSP_NEEDED
%define MEMBERS                 \
    virtual ~Parser() {}        \
    private:                    \
       yyFlexLexer lexer;
%define LEX_BODY {return lexer.yylex();}
%define ERROR_BODY {cerr << "error encountered at line: "<<lexer.lineno()<<" last word parsed:"<<lexer.YYText()<<"\n";}
%define USE_CONST_TOKEN 1

%header{
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string>
#include "types.h"
#include <FlexLexer.h>
using namespace std;

%}

%union {
    char *sValue;                /* string*/
    char *sName;
    char *sParam;
    nodeType *nPtr;             /* node pointer */
    float fValue;                 /* approximate number */
    int iValue;
    char* sSubtok;      /* comparator subtokens */
    int iLength;
};

%token <sName> NAME
%token <sValue> STRING 
%token <iValue> INTNUM
%token <fValue> APPROXNUM 

%left OR
%left AND
%left NOT
%left <sSubtok> COMPARISON /* = <> < > <= >= */
%left '+' '-'
%left '*' '/'
%nonassoc UMINUS

/* literal keyword tokens */

%token ALL BETWEEN BY DISTINCT FROM GROUP HAVING
%token SELECT COMPARISON
%token USER WHERE WITH 
%token EMPTY SELALL DOT 

%token UPDATE SET CURRENT OF NULLX ASSIGN

%token INSERT INTO VALUES

%token CREATE TABLE NOT UNIQUE PRIMARY FOREIGN KEY CHECK REFERENCES DEFAULT DROP
%token DATATYPE
%token DECIMAL SMALLINT NUMERIC CHARACTER INTEGER REAL FLOAT DOUBLE PRECISION VARCHAR

%token AMMSC AVG MAX MIN SUM COUNT

%token ALIAS INTORDER COLORDER AS ORDER ASC DESC
%token LIMIT OFFSET

/*%type <nPtr> sql
%type <nPtr> manipulative_statement
%type <nPtr> select_statement selection table_exp opt_all_distinct
%type <nPtr> from_clause opt_where_clause opt_having_clause column_ref
%type <nPtr> table_ref_commalist scalar_exp_commalist table_ref table

%type <nPtr> search_condition predicate comparison_predicate between_predicate scalar_exp
%type <nPtr> literal atom

%type <nPtr> update_statement_positioned update_statement_searched 
%type <nPtr> assignment assignment_commalist column cursor

%type <nPtr> insert_statement column_commalist opt_column_commalist
%type <nPtr> values_or_query_spec insert_atom_commalist insert_atom query_spec

%type <nPtr> schema base_table_def base_table_element_commalist base_table_element 
%type <nPtr> column_def table_constraint_def column_def_opt column_def_opt_list data_type

%type <nPtr> function_ref ammsc opt_group_by_clause column_ref_commalist 
%type <nPtr> opt_asc_desc opt_order_by_clause ordering_spec ordering_spec_commalist
%type <nPtr> opt_limit_clause*/

%start program

%%

program 
: sql_list                        
;

sql_list
: sql ';'
| sql_list sql ';'
;

opt_column_commalist
:        /* empty */
|   '(' column_commalist ')'
;
    
    /* schema definition language. Basic Sql Parser only uses a fraction of it. */
sql:        
schema
;
    
schema
: base_table_def
;

base_table_def
: CREATE TABLE table '(' base_table_element_commalist ')'
| DROP TABLE table
;

base_table_element_commalist
: base_table_element     
| base_table_element_commalist ',' base_table_element
;

base_table_element
: column_def
| table_constraint_def
;

column_def
: column data_type column_def_opt_list
;

column_def_opt_list
:        /* empty */
| column_def_opt_list column_def_opt
;

column_def_opt
: NOT NULLX
| NOT NULLX UNIQUE
| NOT NULLX PRIMARY KEY
| DEFAULT literal
| DEFAULT NULLX
| DEFAULT USER
| CHECK '(' search_condition ')'
| REFERENCES table
| REFERENCES table '(' column_commalist ')'
;

table_constraint_def
: UNIQUE '(' column_commalist ')'             
| PRIMARY KEY '(' column_commalist ')'        
| FOREIGN KEY '(' column_commalist ')' REFERENCES table                  
| FOREIGN KEY '(' column_commalist ')' REFERENCES table '(' column_commalist ')' 
| CHECK '(' search_condition ')'
;

column_commalist
: column
| column_commalist ',' column
;

opt_order_by_clause
:        /* empty */
| ORDER BY ordering_spec_commalist
;
    
ordering_spec_commalist
: ordering_spec
| ordering_spec_commalist ',' ordering_spec
;

ordering_spec
: INTNUM opt_asc_desc
| column_ref opt_asc_desc
;

opt_asc_desc
:        /* empty */
| ASC
| DESC
;

/* this starts the execution of classic manipulative statements. */

sql
: manipulative_statement
;

manipulative_statement
: select_statement
| update_statement_positioned
| update_statement_searched
| insert_statement
    /*
| commit_statement
| delete_statement_positioned
| delete_statement_searched
| fetch_statement
| insert_statement
| open_statement
| rollback_statement 
| select_statement 
| update_statement_positioned
| update_statement_searched */
;

insert_statement
: INSERT INTO table opt_column_commalist values_or_query_spec
;

values_or_query_spec
: VALUES '(' insert_atom_commalist ')'
| query_spec
;

insert_atom_commalist
: insert_atom
| insert_atom_commalist ',' insert_atom
;

insert_atom
: atom
| NULLX
;

select_statement
: SELECT opt_all_distinct selection
      /*   INTO target_commalist */ 
         table_exp
;

    
opt_all_distinct
: ALL
| DISTINCT
| /* empty  */
;

update_statement_positioned
: UPDATE table SET assignment_commalist       
        WHERE CURRENT OF cursor
;

assignment_commalist
:    /* empty */
| assignment
| assignment_commalist ',' assignment
;

assignment
: column COMPARISON scalar_exp
| column COMPARISON NULLX
;

update_statement_searched
: UPDATE table SET assignment_commalist opt_where_clause
;

query_spec
: SELECT opt_all_distinct selection table_exp
;

selection
: scalar_exp_commalist
| '*'
;

table_exp
: from_clause            
        opt_where_clause
        opt_group_by_clause
        opt_having_clause
        opt_order_by_clause
        opt_limit_clause
; 

from_clause
: FROM table_ref_commalist
;

table_ref_commalist
: table_ref
| table_ref_commalist ',' table_ref
;
    
table_ref
: table
    /* | table rangevariable */
;

opt_where_clause
: WHERE search_condition
| /* empty */
;

opt_group_by_clause
:   /* empty */
| GROUP BY column_ref_commalist
;

column_ref_commalist  
: column_ref
| column_ref_commalist ',' column_ref
;

opt_having_clause
: HAVING search_condition
| /* empty */
;

opt_limit_clause:
    /* empty */                               
    | LIMIT INTNUM                            
    | LIMIT INTNUM ',' INTNUM                 
    | LIMIT INTNUM OFFSET INTNUM              
    /* search conditions */

search_condition:
      search_condition OR search_condition    
    |   search_condition AND search_condition
    |   NOT search_condition
    |   '(' search_condition ')'
    |   predicate
    ;

predicate:
        comparison_predicate
    |   between_predicate
 /*   |   like_predicate
    |   test_for_null
    |   in_predicate
    |   all_or_any_predicate
    |   existence_test */
    ;

comparison_predicate:
        scalar_exp COMPARISON scalar_exp
   /* |   scalar_exp COMPARISON subquery                          { $$ = opr(COMPARISON, $1, $3); } */
    ;

between_predicate:
        scalar_exp NOT BETWEEN scalar_exp AND scalar_exp
    |   scalar_exp BETWEEN scalar_exp AND scalar_exp
    ;

scalar_exp_commalist:       
        scalar_exp
    |   scalar_exp_commalist ',' scalar_exp
    ;

scalar_exp:
    scalar_exp '+' scalar_exp
    | scalar_exp '-' scalar_exp
    | scalar_exp '*' scalar_exp
    | scalar_exp '/' scalar_exp
    |   '+' scalar_exp %prec UMINUS
    |   '-' scalar_exp %prec UMINUS
    |   atom
    |   column_ref
    |   function_ref
    |   '(' scalar_exp ')'
    ;

atom:
    /*  parameter_ref
    | */   literal
    |   USER
    ;
/*
parameter_ref:
        parameter
    |   parameter parameter
    |   parameter INDICATOR parameter
    ;
*/

function_ref:
        ammsc '(' '*' ')'
    |   ammsc '(' DISTINCT column_ref ')'
    |   ammsc '(' ALL scalar_exp ')'
    |   ammsc '(' scalar_exp ')'
    ;

literal:
        STRING
    |   INTNUM
    |   APPROXNUM
    ;

table:
        NAME
        | NAME '.' NAME
        | NAME AS NAME
    ;

/* data types */
data_type:
        CHARACTER
    |   CHARACTER '(' INTNUM ')'
    |   VARCHAR
    |   VARCHAR '(' INTNUM ')'
    |   NUMERIC
    |   NUMERIC '(' INTNUM ')'
    |   NUMERIC '(' INTNUM ',' INTNUM ')'
    |   DECIMAL
    |   DECIMAL '(' INTNUM ')'
    |   DECIMAL '(' INTNUM ',' INTNUM ')'
    |   INTEGER
    |   SMALLINT
    |   FLOAT
    |   FLOAT '(' INTNUM ')'
    |   REAL
    |   DOUBLE PRECISION
    ;

column_ref:
        NAME
    |   NAME '.' NAME
    |   NAME '.' NAME '.' NAME
    |   NAME AS NAME
    ;
        /* the various things you can name */

column:     NAME
    ;

cursor:     NAME
    ;

ammsc: 
    AVG
    | MIN
    | MAX
    | SUM
    | COUNT

%%

