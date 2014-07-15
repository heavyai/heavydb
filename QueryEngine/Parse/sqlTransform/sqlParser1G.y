%{
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include "sqlParser1.h" 

#define YYDEBUG 1

#ifdef DEBUG
#define TRACE printf("reduce at line %d\n", __LINE__);
#else
#define TRACE
#endif

/* prototypes */
nodeType *opr(int oper, int nops, ...);
nodeType *id(char *s);
nodeType *id2(char *s);
nodeType *id3(char* s);
nodeType *text(char *s);
nodeType *comp(char *s);
nodeType *compAssgn(char *s);
nodeType *con(double value);
void freeNode(nodeType *p);
int ex(nodeType *p);
int yylex(void);

int yyerror(const char *s);
%}

%union {
    char *sValue;                /* string*/
    char *sName;
    char *sParam;
    nodeType *nPtr;             /* node pointer */
    double dValue;                 /* approximate number */
    int iValue;
    char* sSubtok;      /* comparator subtokens */
    int iLength;
};

%error-verbose

%token <sName> NAME
%token <sValue> STRING 
%token <iValue> INTNUM
%token <dValue> APPROXNUM "!"

%left OR
%left AND
%left NOT
%left <sSubtok> COMPARISON /* = <> < > <= >= */
%left '+' '-'
%left '*' '/' '.'
%nonassoc UMINUS

/* literal keyword tokens */

%token ALL BETWEEN BY DISTINCT FROM GROUP HAVING
%token SELECT COMPARISON
%token USER WHERE WITH 
%token EMPTY SELALL DOT 

%token UPDATE SET CURRENT OF NULLX ASSIGN

%token INSERT INTO VALUES

%token CREATE NOT UNIQUE PRIMARY FOREIGN KEY CHECK REFERENCES DEFAULT DROP
%token DATATYPE
%token DECIMAL SMALLINT NUMERIC CHARACTER INTEGER REAL FLOAT DOUBLE PRECISION VARCHAR

%token AVG MAX MIN SUM COUNT

%token ALIAS INTORDER COLORDER AS ORDER ASC DESC
%token LIMIT OFFSET

%token DOTNAME

%token SQL
%token  MANIPULATIVE_STATEMENT
%token  SELECT_STATEMENT SELECTION TABLE_EXP OPT_ALL_DISTINCT
%token  FROM_CLAUSE OPT_WHERE_CLAUSE OPT_HAVING_CLAUSE COLUMN_REF
%token  TABLE_REF_COMMALIST SCALAR_EXP_COMMALIST TABLE_REF TABLE
%token  SEARCH_CONDITION PREDICATE COMPARISON_PREDICATE BETWEEN_PREDICATE SCALAR_EXP
%token  LITERAL ATOM
%token  UPDATE_STATEMENT_POSITIONED UPDATE_STATEMENT_SEARCHED 
%token  ASSIGNMENT ASSIGNMENT_COMMALIST COLUMN CURSOR
%token  INSERT_STATEMENT COLUMN_COMMALIST OPT_COLUMN_COMMALIST
%token  VALUES_OR_QUERY_SPEC INSERT_ATOM_COMMALIST INSERT_ATOM QUERY_SPEC
%token  SCHEMA BASE_TABLE_DEF BASE_TABLE_ELEMENT_COMMALIST BASE_TABLE_ELEMENT 
%token  COLUMN_DEF TABLE_CONSTRAINT_DEF COLUMN_DEF_OPT COLUMN_DEF_OPT_LIST DATA_TYPE
%token  FUNCTION_REF AMMSC OPT_GROUP_BY_CLAUSE COLUMN_REF_COMMALIST 
%token  OPT_ASC_DESC OPT_ORDER_BY_CLAUSE ORDERING_SPEC ORDERING_SPEC_COMMALIST
%token  OPT_LIMIT_CLAUSE PRODUCT PROJECT

%type <nPtr> sql
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
%type <nPtr> opt_limit_clause

%%

program: 
    sql_list        { exit(0);}         /* it all starts here */
    ;

sql_list:
        sql ';' { ex($1); freeNode($1); }
    |   sql_list sql ';' { ex($2); freeNode($2); }
    ;

opt_column_commalist:
        /* empty */                     { $$ = opr(OPT_COLUMN_COMMALIST, 1, opr(EMPTY, 0)); }
    |   '(' column_commalist ')'        { $$ = opr(OPT_COLUMN_COMMALIST, 1, $2); }
    ;
    
    /* schema definition language. Basic Sql Parser only uses a fraction of it. */
sql:        schema      { $$ = opr(SQL, 1, $1); }
    ;
    
schema:
       base_table_def       { $$ = opr(SCHEMA, 1, $1); }
    ;

base_table_def:
        CREATE TABLE table '(' base_table_element_commalist ')'    { $$ = opr(BASE_TABLE_DEF, 4, opr(CREATE, 0), opr(TABLE, 0), $3, $5); }
    | DROP TABLE table                                             { $$ = opr(BASE_TABLE_DEF, 3, opr(DROP, 0), opr(TABLE, 0), $3); }
    ;

base_table_element_commalist:
        base_table_element                                      { $$ = opr(BASE_TABLE_ELEMENT_COMMALIST, 1, $1); }
    |   base_table_element_commalist ',' base_table_element     { $$ = opr(BASE_TABLE_ELEMENT_COMMALIST, 3, $1, opr(',', 0), $3); }
    ;

base_table_element:
        column_def              { $$ = opr(BASE_TABLE_ELEMENT, 1, $1); }
    |   table_constraint_def    { $$ = opr(BASE_TABLE_ELEMENT, 1, $1); }
    ;

column_def:
        column data_type column_def_opt_list        { $$ = opr(COLUMN_DEF, 3, $1, $2, $3); }
    ;

column_def_opt_list:
        /* empty */                                 { $$ = opr(COLUMN_DEF_OPT_LIST, 1, opr(EMPTY, 0)); }
    |   column_def_opt_list column_def_opt          { $$ = opr(COLUMN_DEF_OPT_LIST, 2, $1, $2); }
    ;

column_def_opt:
        NOT NULLX                                   { $$ = opr(COLUMN_DEF_OPT, 2, opr(NOT, 0), opr(NULLX, 0)); }
    |   NOT NULLX UNIQUE                            { $$ = opr(COLUMN_DEF_OPT, 3, opr(NOT, 0), opr(NULLX, 0), opr(UNIQUE, 0)); }
    |   NOT NULLX PRIMARY KEY                       { $$ = opr(COLUMN_DEF_OPT, 3, opr(NOT, 0), opr(NULLX, 0), opr(PRIMARY, 0), opr(KEY, 0)); }
    |   DEFAULT literal                             { $$ = opr(COLUMN_DEF_OPT, 2, opr(DEFAULT, 0), $2); }
    |   DEFAULT NULLX                               { $$ = opr(COLUMN_DEF_OPT, 2, opr(DEFAULT, 0), opr(NULLX, 0)); }
    |   DEFAULT USER                                { $$ = opr(COLUMN_DEF_OPT, 2, opr(DEFAULT, 0), opr(USER, 0)); }
    |   CHECK '(' search_condition ')'              { $$ = opr(COLUMN_DEF_OPT, 2, opr(CHECK, 0), $3); }
    |   REFERENCES table                            { $$ = opr(COLUMN_DEF_OPT, 2, opr(REFERENCES, 0), $2); }
    |   REFERENCES table '(' column_commalist ')'   { $$ = opr(COLUMN_DEF_OPT, 3, opr(REFERENCES, 0), $2, $4); }
    ;

table_constraint_def:
        UNIQUE '(' column_commalist ')'             { $$ = opr(TABLE_CONSTRAINT_DEF, 2, opr(UNIQUE, 0), $3); }
    |   PRIMARY KEY '(' column_commalist ')'        { $$ = opr(TABLE_CONSTRAINT_DEF, 3, opr(PRIMARY, 0), opr(KEY, 0), $4); }
    |   FOREIGN KEY '(' column_commalist ')'        
            REFERENCES table                        { $$ = opr(TABLE_CONSTRAINT_DEF, 5, opr(FOREIGN, 0), opr(KEY, 0), $4, opr(REFERENCES, 0), $7); }
    |   FOREIGN KEY '(' column_commalist ')'
            REFERENCES table '(' column_commalist ')'   { $$ = opr(TABLE_CONSTRAINT_DEF, 6, opr(FOREIGN, 0), opr(KEY, 0), $4, opr(REFERENCES, 0), $7, $9); }
    |   CHECK '(' search_condition ')'              { $$ = opr(TABLE_CONSTRAINT_DEF, 2, opr(CHECK, 0), $3); }
    ;

column_commalist:
        column                                      { $$ = opr(COLUMN_COMMALIST, 1, $1); }
    |   column_commalist ',' column                 { $$ = opr(COLUMN_COMMALIST, 3, $1, opr(',', 0), $3); }
    ;

opt_order_by_clause:
        /* empty */                             { $$ = opr(OPT_ORDER_BY_CLAUSE, 1, opr(EMPTY, 0)); }
    |   ORDER BY ordering_spec_commalist        { $$ = opr(OPT_ORDER_BY_CLAUSE, 3, opr(ORDER, 0), opr(BY, 0), $3); }
    ;
    
ordering_spec_commalist:
        ordering_spec                                   { $$ = opr(ORDERING_SPEC_COMMALIST, 1, $1); }
    |   ordering_spec_commalist ',' ordering_spec       { $$ = opr(ORDERING_SPEC_COMMALIST, 3, $1, ',', $3); }
    ;

ordering_spec:
        INTNUM opt_asc_desc             { $$ = opr(ORDERING_SPEC, 2, con($1), $2); }
    |   column_ref opt_asc_desc         { $$ = opr(ORDERING_SPEC, 2, $1, $2); }
    ;

opt_asc_desc:
        /* empty */                     { $$ = opr(OPT_ASC_DESC, 1, opr(EMPTY, 0)); }
    |   ASC                             { $$ = opr(OPT_ASC_DESC, 1, opr(ASC, 0)); }
    |   DESC                            { $$ = opr(OPT_ASC_DESC, 1, opr(DESC, 0)); }
    ;

/* this starts the execution of classic manipulative statements. */

sql:
        manipulative_statement      { $$ = opr(SQL, 1, $1); }
    ;

manipulative_statement:
        select_statement                    { $$ = opr(MANIPULATIVE_STATEMENT, 1, $1); } 
    |   update_statement_positioned         { $$ = opr(MANIPULATIVE_STATEMENT, 1, $1); } 
    |   update_statement_searched           { $$ = opr(MANIPULATIVE_STATEMENT, 1, $1); } 
    |   insert_statement                    { $$ = opr(MANIPULATIVE_STATEMENT, 1, $1); }
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
        INSERT INTO table opt_column_commalist values_or_query_spec             { $$ = opr(INSERT_STATEMENT, 5, opr(INSERT, 0), opr(INTO, 0), $3, $4, $5); }
    ;

values_or_query_spec:
        VALUES '(' insert_atom_commalist ')'        { $$ = opr(VALUES_OR_QUERY_SPEC, 2, opr(VALUES, 0), $3); }
    |   query_spec                                  { $$ = opr(VALUES_OR_QUERY_SPEC, 1, $1); }         
    ;

insert_atom_commalist:
        insert_atom                                 { $$ = opr(INSERT_ATOM_COMMALIST, 1, $1); } 
    |   insert_atom_commalist ',' insert_atom       { $$ = opr(INSERT_ATOM_COMMALIST, 3, $1, opr(',', 0), $3); }
    ;

insert_atom:
        atom                                        { $$ = opr(INSERT_ATOM, 1, $1); }
    |   NULLX                                       { $$ = opr(INSERT_ATOM, 1, opr(NULLX, 0)); }
    ;

select_statement:
        SELECT opt_all_distinct selection
      /*   INTO target_commalist */ 
         table_exp                                  { $$ = opr(SELECT_STATEMENT, 4, opr(SELECT, 0), $2, $3, $4); }   
    ;

    
opt_all_distinct:
    ALL                                     { $$ = opr(OPT_ALL_DISTINCT, 1, opr(ALL, 0)); }
    |   DISTINCT                            { $$ = opr(OPT_ALL_DISTINCT, 1, opr(DISTINCT, 0)); }
    | /* empty  */                          { $$ = opr(OPT_ALL_DISTINCT, 1, opr(EMPTY, 0)); }
    ;

update_statement_positioned:
        UPDATE table SET assignment_commalist       
        WHERE CURRENT OF cursor                 { $$ = opr(UPDATE_STATEMENT_POSITIONED, 8, opr(UPDATE, 0), $2, opr(SET, 0), $4,
                                                       opr(WHERE, 0), opr(CURRENT, 0), opr(OF, 0), $8); }
    ;

assignment_commalist:
         /* empty */                            { $$ = opr(ASSIGNMENT_COMMALIST, 1, opr(EMPTY, 0)); }
    |   assignment                              { $$ = opr(ASSIGNMENT, 1, $1); }
    |   assignment_commalist ',' assignment     { $$ = opr(ASSIGNMENT_COMMALIST, 3, $1, opr(',', 0), $3); }
    ;

assignment:
        column COMPARISON scalar_exp               { $$ = opr(ASSIGNMENT, 3, $1, compAssgn($2), $3); }
    |   column COMPARISON NULLX                    { $$ = opr(ASSIGNMENT, 3, $1, compAssgn($2), opr(NULLX, 0)); }
    ;

update_statement_searched:
        UPDATE table SET assignment_commalist opt_where_clause      { $$ = opr(UPDATE_STATEMENT_SEARCHED, 5, opr(UPDATE, 0), $2, opr(SET, 0), $4, $5); }
    ;

query_spec:
        SELECT opt_all_distinct selection table_exp     { $$ = opr(SELECT_STATEMENT, 4, opr(SELECT, 0), $2, $3, $4); }
    ;

selection:
    scalar_exp_commalist         { $$ = opr(SELECTION, 1, $1); } 
    |   '*'                     { $$ = opr(SELECTION, 1, opr(SELALL, 0)); }
    ;

table_exp:
        from_clause            
        opt_where_clause
        opt_group_by_clause
        opt_having_clause
        opt_order_by_clause
        opt_limit_clause       { $$ = opr(TABLE_EXP, 6, $1, $2, $3, $4, $5, $6); }
         ; 

from_clause:
        FROM table_ref_commalist        { $$ = opr(FROM_CLAUSE, 2, opr(FROM, 0), $2); }
    ;

table_ref_commalist:
        table_ref                           { $$ = opr(TABLE_REF_COMMALIST, 1, $1); }
    |   table_ref_commalist ',' table_ref    { $$ = opr(TABLE_REF_COMMALIST, 3, $1, opr(',', 0), $3); }
    ;
    
table_ref:
    table                                    { $$ = opr(TABLE_REF, 1, $1); }
    /* | table rangevariable */
    ;

opt_where_clause:
    WHERE search_condition                      { $$ = opr(OPT_WHERE_CLAUSE, 2, opr(WHERE, 0), $2); }
    | /* empty */                               { $$ = opr(OPT_WHERE_CLAUSE, 1, opr(EMPTY, 0)); }
    ;

opt_group_by_clause:
        /* empty */                             { $$ = opr(OPT_GROUP_BY_CLAUSE, 1, opr(EMPTY, 0)); }
    |   GROUP BY column_ref_commalist           { $$ = opr(OPT_GROUP_BY_CLAUSE, 3, opr(GROUP, 0), opr(BY, 0), $3); }
    ;

column_ref_commalist:   
        column_ref                              { $$ = opr(COLUMN_REF_COMMALIST, 1, $1); }
    |   column_ref_commalist ',' column_ref     { $$ = opr(COLUMN_REF_COMMALIST, 3, $1, opr(',', 0), $3); }
    ;

opt_having_clause:
    HAVING search_condition                 { $$ = opr(OPT_HAVING_CLAUSE, 2, opr(HAVING, 0), $2); }
    | /* empty */                           { $$ = opr(OPT_HAVING_CLAUSE, 1, opr(EMPTY, 0)); }
    ;

opt_limit_clause:
    /* empty */                               { $$ = opr(OPT_LIMIT_CLAUSE, 1, opr(EMPTY, 0)); }
    | LIMIT INTNUM                            { $$ = opr(OPT_LIMIT_CLAUSE, 2, opr(LIMIT, 0), con($2)); }
    | LIMIT INTNUM ',' INTNUM                 { $$ = opr(OPT_LIMIT_CLAUSE, 3, opr(LIMIT, 0), opr(',', 0), con($2)); }
    | LIMIT INTNUM OFFSET INTNUM              { $$ = opr(OPT_LIMIT_CLAUSE, 3, opr(LIMIT, 0), opr(OFFSET, 0), con($2)); }
    /* search conditions */

search_condition: 
      search_condition OR search_condition      { $$ = opr(SEARCH_CONDITION, 3, $1, opr(OR, 0), $3); }
    |   search_condition AND search_condition   { $$ = opr(SEARCH_CONDITION, 3, $1, opr(AND, 0), $3); }
    |   NOT search_condition                    { $$ = opr(SEARCH_CONDITION, 2, opr(NOT, 0), $2); }
    |   '(' search_condition ')'                { $$ = opr(SEARCH_CONDITION, 1, $2); }
    |   predicate                               { $$ = opr(SEARCH_CONDITION, 1, $1); }
    ;

predicate:
        comparison_predicate                    { $$ = opr(PREDICATE, 1, $1); }
    |   between_predicate                       { $$ = opr(PREDICATE, 1, $1); }
 /*   |   like_predicate
    |   test_for_null
    |   in_predicate
    |   all_or_any_predicate
    |   existence_test */
    ;

comparison_predicate:
        scalar_exp COMPARISON scalar_exp                        { $$ = opr(COMPARISON_PREDICATE, 3, $1, comp($2), $3); }
   /* |   scalar_exp COMPARISON subquery                          { $$ = opr(COMPARISON, $1, $3); } */
    ;

between_predicate:
        scalar_exp NOT BETWEEN scalar_exp AND scalar_exp        { $$ = opr(BETWEEN_PREDICATE, 6, $1, opr(NOT, 0), opr(BETWEEN, 0), $4, opr(AND, 0), $6); }
    |   scalar_exp BETWEEN scalar_exp AND scalar_exp            { $$ = opr(BETWEEN_PREDICATE, 5, $1, opr(BETWEEN, 0), $3, opr(AND, 0), $5); }
    ;

scalar_exp_commalist:       
        scalar_exp                              { $$ = opr(SCALAR_EXP_COMMALIST, 1, $1); }
    |   scalar_exp_commalist ',' scalar_exp     { $$ = opr(SCALAR_EXP_COMMALIST, 3, $1, opr(',', 0), $3); }
    ;

scalar_exp:
    scalar_exp '+' scalar_exp               { $$ = opr(SCALAR_EXP, 3, $1, opr('+', 0), $3); }
    | scalar_exp '-' scalar_exp             { $$ = opr(SCALAR_EXP, 3, $1, opr('-', 0), $3); }
    | scalar_exp '*' scalar_exp             { $$ = opr(SCALAR_EXP, 3, $1, opr('*', 0), $3); }
    | scalar_exp '/' scalar_exp             { $$ = opr(SCALAR_EXP, 3, $1, opr('/', 0), $3); }
    |   '+' scalar_exp %prec UMINUS         { $$ = opr(SCALAR_EXP, 3, opr('+', 0), $2, opr(UMINUS, 0)); }
    |   '-' scalar_exp %prec UMINUS         { $$ = opr(SCALAR_EXP, 3, opr('-', 0), $2, opr(UMINUS, 0)); }
    |   atom                                { $$ = opr(SCALAR_EXP, 1, $1); }
    |   column_ref                          { $$ = opr(SCALAR_EXP, 1, $1); }
    |   function_ref                        { $$ = opr(SCALAR_EXP, 1, $1); }
    |   '(' scalar_exp ')'                  { $$ = opr(SCALAR_EXP, 1, $2); }
    ;

atom:
    /*  parameter_ref
    | */   literal           { $$ = opr(ATOM, 1, $1); }
    |   USER                { $$ = opr(ATOM, 1, opr(USER, 0)); }
    ;
/*
parameter_ref:
        parameter               { $$ = $1; }
    |   parameter parameter                 { $$ = opr }
    |   parameter INDICATOR parameter
    ;
*/

function_ref:
        ammsc '(' '*' ')'                   { $$ = opr(FUNCTION_REF, 2, $1, opr(SELALL, 0));}
    |   ammsc '(' DISTINCT column_ref ')'   { $$ = opr(FUNCTION_REF, 3, $1, opr(DISTINCT, 0), $4); }
    |   ammsc '(' ALL scalar_exp ')'        { $$ = opr(FUNCTION_REF, 3, $1, opr(ALL, 0), $4); }
    |   ammsc '(' scalar_exp ')'            { $$ = opr(FUNCTION_REF, 2, $1, $3); }
    ;

literal:
        STRING                              { $$ = opr(LITERAL, 1, text($1)); }
    |   INTNUM                              { $$ = opr(LITERAL, 1, con($1)); }
    |   APPROXNUM                           { $$ = opr(LITERAL, 1, con($1)); }
    ;
;

table:
        NAME                { $$ = opr(TABLE, 1, id($1)); }
        | NAME '.' NAME     { $$ = opr(TABLE, 3, id($1), opr('.', 0), id2($3));}
        | NAME AS NAME      { $$ = opr(TABLE, 3, id($1), opr(AS, 0), id2($3));  }
    ;

/* data types */
data_type:
        CHARACTER                           { $$ = opr(DATATYPE, 1, opr(CHARACTER, 0)); }
    |   CHARACTER '(' INTNUM ')'            { $$ = opr(DATATYPE, 2, con($3)); }
    |   VARCHAR                             { $$ = opr(DATATYPE, 1, opr(VARCHAR, 0)); }
    |   VARCHAR '(' INTNUM ')'              { $$ = opr(DATATYPE, 2, con($3)); }
    |   NUMERIC                             { $$ = opr(DATATYPE, 1, opr(NUMERIC, 0)); }
    |   NUMERIC '(' INTNUM ')'              { $$ = opr(DATATYPE, 2, con($3)); }
    |   NUMERIC '(' INTNUM ',' INTNUM ')'   { $$ = opr(DATATYPE, 2, con($3)); }
    |   DECIMAL                             { $$ = opr(DATATYPE, 1, opr(DECIMAL, 0)); }
    |   DECIMAL '(' INTNUM ')'              { $$ = opr(DATATYPE, 2, con($3)); }
    |   DECIMAL '(' INTNUM ',' INTNUM ')'   { $$ = opr(DATATYPE, 2, con($3)); }
    |   INTEGER                             { $$ = opr(DATATYPE, 1, opr(INTEGER, 0)); }
    |   SMALLINT                            { $$ = opr(DATATYPE, 1, opr(SMALLINT, 0)); }
    |   FLOAT                               { $$ = opr(DATATYPE, 1, opr(FLOAT, 0)); }
    |   FLOAT '(' INTNUM ')'                { $$ = opr(DATATYPE, 2, opr(FLOAT, 0), con($3)); }
    |   REAL                                { $$ = opr(DATATYPE, 1, opr(REAL, 0)); }
    |   DOUBLE PRECISION                    { $$ = opr(DATATYPE, 2, opr(DOUBLE, 0), opr(PRECISION, 0)); }

column_ref:
        NAME                    { $$ = opr(COLUMN_REF, 1, id($1)); }
    |   NAME '.' NAME           { $$ = opr(COLUMN_REF, 3, id($1), opr('.', 0), id2($3));}
    |   NAME '.' NAME '.' NAME  { $$ = opr(COLUMN_REF, 5, id($1), opr('.', 0), id2($3), opr('.', 0), id3($5));}
    |   NAME AS NAME            { $$ = opr(COLUMN_REF, 3, id($1), opr(AS, 0), id2($3)); } 
    ;
        /* the various things you can name */

column:     NAME            { $$ = opr(COLUMN, 1, id($1)); }
    ;

cursor:     NAME            { $$ = opr(CURSOR, 1, id($1)); }
    ;

ammsc: 
    AVG             { $$ = opr(AMMSC, 1, opr(AVG, 0)); }
    | MIN           { $$ = opr(AMMSC, 1, opr(MIN, 0)); }
    | MAX           { $$ = opr(AMMSC, 1, opr(MAX, 0)); }
    | SUM           { $$ = opr(AMMSC, 1, opr(SUM, 0)); }
    | COUNT         { $$ = opr(AMMSC, 1, opr(COUNT, 0)); }

%%

nodeType *id(char* s) {
    nodeType *p;

    /* allocate node */
    if ((p = malloc(sizeof(nodeType))) == NULL)
        yyerror("out of memory");

    /* copy information */
    p->type = typeId;
    p->id.s = s;
    p->id.iLength = textLength;
    //printf("text: %d %s \n", textLength, s);
    return p;
}

/* backup of id when two names are needed simultaneously, such as Name.Name */
nodeType *id2(char* s) {
    nodeType *p;

    /* allocate node */
    if ((p = malloc(sizeof(nodeType))) == NULL)
        yyerror("out of memory");

    /* copy information */
    p->type = typeId;
    p->id.s = s;
    p->id.iLength = textLength2;
    //printf("text: %d %s \n", textLength2, s);
    return p;
}

/* backup of id2 when three names are needed simultaneously, such as Name.Name.Name */
nodeType *id3(char* s) {
    nodeType *p;

    /* allocate node */
    if ((p = malloc(sizeof(nodeType))) == NULL)
        yyerror("out of memory");

    /* copy information */
    p->type = typeId;
    p->id.s = s;
    p->id.iLength = textLength3;
    //Fprintf("text: %d %s \n", textLength2, s);
    return p;
}

/* Handles regular text, as copied by the STRING token. */
nodeType *text(char* s) {
    nodeType *p;

    /* allocate node */
    if ((p = malloc(sizeof(nodeType))) == NULL)
        yyerror("out of memory");

    /* copy information */
    p->type = typeText;
    p->id.s = s;
    p->id.iLength = textLength;
 //   printf("text: %d %s \n", textLength, s);
    return p;
}


nodeType *comp(char* s) {
    nodeType *p;

    /* comparators: =, >, etc. */
    /* allocate node */
    if ((p = malloc(sizeof(nodeType))) == NULL)
        yyerror("out of memory");

    /* copy information */
    p->type = typeComp;
    p->id.s = s;
    p->id.iLength = comparisonLength;
   // printf("comp: %d %s \n", comparisonLength, s);
    return p;
}

/* Treat assignment statements as a comparator. Cheesy, I know. */
nodeType *compAssgn(char* s) {
    nodeType *p;

    /* Treat all comparators: =, >, etc. that appear grammatically as an assignment "=" as that "=".

    To do: Find a workaround that allows the assignment rule to appear properly in the grammar. */
    /* allocate node */
    if ((p = malloc(sizeof(nodeType))) == NULL)
        yyerror("out of memory");

    if (!strcmp(s, "=")) {
        printf("wrong comparator\n");
        fflush(stdout);
        yyerror("wrong comparator");
    }
    /* copy information */
    p->type = typeAssgn;
    p->id.s = "=";
    p->id.iLength = strlen("=");
  //  printf("comp: %d %s \n", comparisonLength, s);
    return p;
}

nodeType *con(double value) {
    nodeType *p;

    /* allocate node */
    if ((p = malloc(sizeof(nodeType))) == NULL)
        yyerror("out of memory");

    /* copy information */
    p->type = typeCon;
    p->con.dValue = value;

    return p;
}

nodeType *opr(int oper, int nops, ...) {
    va_list ap;
    nodeType *p;
    int i;

    /* allocate node */
    if ((p = malloc(sizeof(nodeType))) == NULL)
        yyerror("out of memory");
    if ((p->opr.op = malloc(nops * sizeof(nodeType))) == NULL)
        yyerror("out of memory");

    /* copy information */
    p->type = typeOpr;
    p->opr.oper = oper;
    p->opr.nops = nops;
    va_start(ap, nops);
    for (i = 0; i < nops; i++)
        p->opr.op[i] = va_arg(ap, nodeType*);
    va_end(ap);
    return p;
}

void freeNode(nodeType *p) {
    int i;

    if (!p) return;
    if (p->type == typeOpr) {
        for (i = 0; i < p->opr.nops; i++)
            freeNode(p->opr.op[i]);
		free (p->opr.op);
    }
    free (p);
}

int yyerror(const char *s) {
    fprintf(stderr, "%s\n", s);
    /* should this return 1? */
    return 1;
}

int main(void) {
    int i = yyparse();
    //printf("success? %d\n", i);
    return i;
}