%name Parser
%define LSP_NEEDED
%define MEMBERS                 \
    virtual ~Parser()   {} \
    private:                   \
       yyFlexLexer lexer;
%define LEX_BODY {return lexer.yylex();}
%define USE_CONST_TOKEN 1

%header{
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include "types.h"
#include <FlexLexer.h>

//#define YYDEBUG 1

//#ifdef DEBUG
//#define TRACE printf("reduce at line %d\n", __LINE__);
//#else
//#define TRACE
//#endif

/* prototypes */
nodeType *opr(int oper, int nops, ...);
nodeType *id(char *s);
nodeType *id2(char *s);
nodeType *text(char *s);
nodeType *comp(char *s);
nodeType *compAssgn(char *s);
nodeType *con(float value);
void freeNode(nodeType *p);
int ex(nodeType *p);
//int yylex(void);

//int yyerror(const char *s);

extern int readInputForLexer(char* buffer,int *numBytesRead,int maxBytesToRead);
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
    |   sql_list sql ';' { freeNode($2); }
    ;

opt_column_commalist:
        /* empty */                     { $$ = opr(EMPTY, 0); }
    |   '(' column_commalist ')'        { $$ = $2; }
    ;
    
    /* schema definition language. Basic Sql Parser only uses a fraction of it. */
sql:        schema      { $$ = $1; }
    ;
    
schema:
       base_table_def       { $$ = $1; }
    ;

base_table_def:
        CREATE TABLE table '(' base_table_element_commalist ')'    { $$ = opr(CREATE, 2, opr(TABLE, 1, $3), $5); }
    | DROP TABLE table                                             { $$ = opr(DROP, 1, opr(TABLE, 1, $3)); }
    ;

base_table_element_commalist:
        base_table_element                                      { $$ = $1; }
    |   base_table_element_commalist ',' base_table_element     { $$ = opr(',', 2, $1, $3); }
    ;

base_table_element:
        column_def              { $$ = $1; }
    |   table_constraint_def    { $$ = $1; }
    ;

column_def:
        column data_type column_def_opt_list        { $$ = opr(DATATYPE, 3, $1, $2, $3); }
    ;

column_def_opt_list:
        /* empty */                                 { $$ = opr(EMPTY, 0); }
    |   column_def_opt_list column_def_opt          { $$ = opr(',', 2, $1, $2); }
    ;

column_def_opt:
        NOT NULLX                                   { $$ = opr(NOT, 1, opr(NULLX, 0)); }
    |   NOT NULLX UNIQUE                            { $$ = opr(UNIQUE, 1, opr(NOT, 1, opr(NULLX, 0))); }
    |   NOT NULLX PRIMARY KEY                       { $$ = opr(KEY, 1, opr(PRIMARY, 1, opr(NOT, 1, opr(NULLX, 0)))); }
    |   DEFAULT literal                             { $$ = opr(DEFAULT, 1, $2); }
    |   DEFAULT NULLX                               { $$ = opr(DEFAULT, 1, opr(NULLX, 0)); }
    |   DEFAULT USER                                { $$ = opr(DEFAULT, 1, opr(USER, 0)); }
    |   CHECK '(' search_condition ')'              { $$ = opr(CHECK, 1, $3); }
    |   REFERENCES table                            { $$ = opr(REFERENCES, 1, $2); }
    |   REFERENCES table '(' column_commalist ')'   { $$ = opr(REFERENCES, 2, $2, $4); }
    ;

table_constraint_def:
        UNIQUE '(' column_commalist ')'             { $$ = opr(UNIQUE, 1, $3); }
    |   PRIMARY KEY '(' column_commalist ')'        { $$ = opr(KEY, 1, opr(PRIMARY, 1, $4)); }
    |   FOREIGN KEY '(' column_commalist ')'        
            REFERENCES table                        { $$ = opr(KEY, 2, opr(FOREIGN, 1, $4), opr(REFERENCES, 1, $7)); }
    |   FOREIGN KEY '(' column_commalist ')'
            REFERENCES table '(' column_commalist ')'   { $$ = opr(KEY, 2, opr(FOREIGN, 1, $4), opr(REFERENCES, 2, $7, $9)); }
    |   CHECK '(' search_condition ')'              { $$ = opr(CHECK, 1, $3); }
    ;

column_commalist:
        column                                      { $$ = $1; }
    |   column_commalist ',' column                 { $$ = opr(',', 2, $1, $3); }
    ;

opt_order_by_clause:
        /* empty */                             { $$ = opr(EMPTY, 0); }
    |   ORDER BY ordering_spec_commalist        { $$ = opr(ORDER, 1, opr(BY, 1, $3)); }
    ;
    
ordering_spec_commalist:
        ordering_spec                                   { $$ = $1; }
    |   ordering_spec_commalist ',' ordering_spec       { $$ = opr(',', 2, $1, $3); }
    ;

ordering_spec:
        INTNUM opt_asc_desc             { $$ = opr(INTORDER, 2, con($1), $2); }
    |   column_ref opt_asc_desc         { $$ = opr(COLORDER, 2, $1, $2); }
    ;

opt_asc_desc:
        /* empty */                     { $$ = opr(EMPTY, 0); }
    |   ASC                             { $$ = opr(ASC, 0); }
    |   DESC                            { $$ = opr(DESC, 0); }
    ;

/* this starts the execution of classic manipulative statements. */

sql:
        manipulative_statement      { $$ = $1; }
    ;

manipulative_statement:
        select_statement         { $$ = $1; } 
    |   update_statement_positioned { $$ = $1; } 
    |   update_statement_searched { $$ = $1; } 
    |   insert_statement { $$ = $1; }
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
        INSERT INTO table opt_column_commalist values_or_query_spec             { $$ = opr(INSERT, 3, opr(INTO, 1, $3), $4, $5); }
    ;

values_or_query_spec:
        VALUES '(' insert_atom_commalist ')'        { $$ = opr(VALUES, 1, $3); }
    |   query_spec                                  { $$ = $1; }         
    ;

insert_atom_commalist:
        insert_atom                                 { $$ = $1; } 
    |   insert_atom_commalist ',' insert_atom       { $$ = opr(',', 2, $1, $3); }
    ;

insert_atom:
        atom                                        { $$ = $1; }
    |   NULLX                                       { $$ = NULL; }
    ;

select_statement:
        SELECT opt_all_distinct selection
      /*   INTO target_commalist */ 
         table_exp                           { $$ = opr(SELECT, 3, $2, $3, $4); }     
    ;

    
opt_all_distinct:
    ALL                                     { $$ = opr(ALL, 0); }
    |   DISTINCT                            { $$ = opr(DISTINCT, 0); }
    | /* empty  */                          { $$ = opr(EMPTY, 0); }
    ;

update_statement_positioned:
        UPDATE table SET assignment_commalist       
        WHERE CURRENT OF cursor                 { $$ = opr(UPDATE, 5, $2, opr(SET, 1, $4), opr(WHERE, 0), opr(CURRENT, 0), opr(OF, 1, $8)); }
    ;

assignment_commalist:
         /* empty */       { $$ = opr(EMPTY, 0); }
    |   assignment                              { $$ = $1;}
    |   assignment_commalist ',' assignment     { $$ = opr(',', 2, $1, $3); }
    ;

assignment:
        column COMPARISON scalar_exp               { $$ = opr(ASSIGN, 3, $1, compAssgn($2), $3); }
    |   column COMPARISON NULLX                    { $$ = opr(ASSIGN, 3, $1, compAssgn($2), opr(NULLX, 0)); }
    ;

update_statement_searched:
        UPDATE table SET assignment_commalist opt_where_clause { $$ = opr(UPDATE, 3, $2, opr(SET, 1, $4), $5); }
    ;

query_spec:
        SELECT opt_all_distinct selection table_exp     { $$ = opr(SELECT, 3, $2, $3, $4); }
    ;

selection:
    scalar_exp_commalist         { $$ = $1; } 
    |   '*'                     { $$ = opr(SELALL, 0); }
    ;

table_exp:
        from_clause            
        opt_where_clause
        opt_group_by_clause
        opt_having_clause
        opt_order_by_clause
        opt_limit_clause       { $$ = opr(FROM, 6, $1, $2, $3, $4, $5, $6); }
         ; 

from_clause:
        FROM table_ref_commalist        { $$ = $2; }
    ;

table_ref_commalist:
        table_ref                           { $$ = $1; }
    |   table_ref_commalist ',' table_ref    { $$ = opr(',', 2, $1, $3); }
    ;
    
table_ref:
    table                                    { $$ = $1; }
    /* | table rangevariable */
    ;

opt_where_clause:
    WHERE search_condition                      { $$ = opr(WHERE, 1, $2); }
    | /* empty */                               { $$ = opr(EMPTY, 0); }
    ;

opt_group_by_clause:
        /* empty */                             { $$ = opr(EMPTY, 0); }
    |   GROUP BY column_ref_commalist           { $$ = opr(GROUP, 1, opr(BY, 1, $3)); }
    ;

column_ref_commalist:   
        column_ref                              { $$ = $1; }
    |   column_ref_commalist ',' column_ref     { $$ = opr(',', 2, $1, $3); }
    ;

opt_having_clause:
    HAVING search_condition                 { $$ = opr(HAVING, 1, $2); }
    | /* empty */                           { $$ = opr(EMPTY, 0); }
    ;

opt_limit_clause:
    /* empty */                               { $$ = opr(EMPTY, 0); }
    | LIMIT INTNUM                            { $$ = opr(LIMIT, 1, con($2)); }
    | LIMIT INTNUM ',' INTNUM                 { $$ = opr(LIMIT, 2, con($2), con($4)); }
    | LIMIT INTNUM OFFSET INTNUM              { $$ = opr(LIMIT, 2, con($2), con($4)); }
    /* search conditions */

search_condition:
      search_condition OR search_condition    { $$ = opr(OR, 2, $1, $3); }
    |   search_condition AND search_condition   { $$ = opr(AND, 2, $1, $3); }
    |   NOT search_condition                    { $$ = opr(NOT, 1, $2); }
    |   '(' search_condition ')'                { $$ = $2; }
    |   predicate                               { $$ = $1; }
    ;

predicate:
        comparison_predicate                    { $$ = $1; }
    |   between_predicate                       { $$ = $1; }
 /*   |   like_predicate
    |   test_for_null
    |   in_predicate
    |   all_or_any_predicate
    |   existence_test */
    ;

comparison_predicate:
        scalar_exp COMPARISON scalar_exp                        { $$ = opr(COMPARISON, 3, $1, comp($2), $3); }
   /* |   scalar_exp COMPARISON subquery                          { $$ = opr(COMPARISON, $1, $3); } */
    ;

between_predicate:
        scalar_exp NOT BETWEEN scalar_exp AND scalar_exp        { $$ = opr(NOT, 1, opr(BETWEEN, 2, $1, opr(AND, 2, $4, $6))); }
    |   scalar_exp BETWEEN scalar_exp AND scalar_exp            { $$ = opr(BETWEEN, 2, $1, opr(AND, 2, $3, $5)); }
    ;

scalar_exp_commalist:       
        scalar_exp                              { $$ = $1; }
    |   scalar_exp_commalist ',' scalar_exp     { $$ = opr(',', 2, $1, $3); }
    ;

scalar_exp:
    scalar_exp '+' scalar_exp            { $$ = opr('+', 2, $1, $3); }
    | scalar_exp '-' scalar_exp            { $$ = opr('-', 2, $1, $3); }
    | scalar_exp '*' scalar_exp            { $$ = opr('*', 2, $1, $3); }
    | scalar_exp '/' scalar_exp            { $$ = opr('/', 2, $1, $3); }
    |   '+' scalar_exp %prec UMINUS        { $$ = $2; }
    |   '-' scalar_exp %prec UMINUS        { $$ = opr(UMINUS, 1, $2); }
    |   atom                               { $$ = $1; }
    |   column_ref                          { $$ = $1; }
    |   function_ref                        { $$ = $1; }
    |   '(' scalar_exp ')'                 { $$ = $2; }
    ;

atom:
    /*  parameter_ref
    | */   literal           { $$ = $1; }
    |   USER                { $$ = opr(USER, 0); }
    ;
/*
parameter_ref:
        parameter               { $$ = $1; }
    |   parameter parameter                 { $$ = opr }
    |   parameter INDICATOR parameter
    ;
*/

function_ref:
        ammsc '(' '*' ')'                   { $$ = opr(AMMSC, 2, $1, opr(SELALL, 0));}
    |   ammsc '(' DISTINCT column_ref ')'   { $$ = opr(AMMSC, 3, $1, opr(DISTINCT, 0), $4); }
    |   ammsc '(' ALL scalar_exp ')'        { $$ = opr(AMMSC, 3, $1, opr(ALL, 0), $4); }
    |   ammsc '(' scalar_exp ')'            { $$ = opr(AMMSC, 2, $1, $3); }
    ;

literal:
        STRING          { $$ = text($1); }
    |   INTNUM          { $$ = con($1); }
    |   APPROXNUM       { $$ = con($1); }
    ;

table:
        NAME                { $$ = id($1); }
        | NAME '.' NAME     { $$ = opr(DOT, 2, id($1), id2($3));}
        | NAME AS NAME      { $$ = opr(ALIAS, 2, id($1), id2($3));  }
    ;

/* data types */
data_type:
        CHARACTER                           { $$ = opr(CHARACTER, 0); }
    |   CHARACTER '(' INTNUM ')'            { $$ = opr(CHARACTER, 1, con($3)); }
    |   VARCHAR                           { $$ = opr(VARCHAR, 0); }
    |   VARCHAR '(' INTNUM ')'            { $$ = opr(VARCHAR, 1, con($3)); }
    |   NUMERIC                             { $$ = opr(NUMERIC, 0); }
    |   NUMERIC '(' INTNUM ')'              { $$ = opr(NUMERIC, 1, con($3)); }
    |   NUMERIC '(' INTNUM ',' INTNUM ')'   { $$ = opr(NUMERIC, 1, opr(',', 2, $3, $5)); }
    |   DECIMAL                             { $$ = opr(DECIMAL, 0); }
    |   DECIMAL '(' INTNUM ')'              { $$ = opr(DECIMAL, 1, con($3)); }
    |   DECIMAL '(' INTNUM ',' INTNUM ')'   { $$ = opr(DECIMAL, 1, opr(',', 2, con($3), con($5))); }
    |   INTEGER                             { $$ = opr(INTEGER, 0); }
    |   SMALLINT                            { $$ = opr(SMALLINT, 0); }
    |   FLOAT                               { $$ = opr(FLOAT, 0); }
    |   FLOAT '(' INTNUM ')'                { $$ = opr(FLOAT, 1, con($3)); }
    |   REAL                                { $$ = opr(REAL, 0); }
    |   DOUBLE PRECISION                    { $$ = opr(DOUBLE, 1, opr(PRECISION, 0)); }
    ;

column_ref:
        NAME                { $$ = id($1); }
    |   NAME '.' NAME       { $$ = opr(DOT, 2, id($1), id2($3));}
    |   NAME '.' NAME '.' NAME  { $$ = opr(DOT, 2, id($1),  opr(DOT, 2, id($1), id2($3)));}
    |   NAME AS NAME            { $$ = opr(ALIAS, 2, id($1), id2($3)); } 
    ;
        /* the various things you can name */

column:     NAME            { $$ = id($1); }
    ;

cursor:     NAME            { $$ = id($1); }
    ;

ammsc: 
    AVG             { $$ = opr(AVG, 0); }
    | MIN           { $$ = opr(MIN, 0); }
    | MAX           { $$ = opr(MAX, 0); }
    | SUM           { $$ = opr(SUM, 0); }
    | COUNT         { $$ = opr(COUNT, 0); }

%%


nodeType *id(char* s) {
    nodeType *p;

    /* allocate node */
    if ((p = (nodeType*)malloc(sizeof(nodeType))) == NULL)
        printf("out of memory");

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
    if ((p = (nodeType*)malloc(sizeof(nodeType))) == NULL)
        printf("out of memory");

    /* copy information */
    p->type = typeId;
    p->id.s = s;
    p->id.iLength = textLength2;
    //Fprintf("text: %d %s \n", textLength2, s);
    return p;
}

/* Handles regular text, as copied by the STRING token. */
nodeType *text(char* s) {
    nodeType *p;

    /* allocate node */
    if ((p = (nodeType*)malloc(sizeof(nodeType))) == NULL)
        printf("out of memory");

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
    if ((p = (nodeType*)malloc(sizeof(nodeType))) == NULL)
        printf("out of memory");

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
    if ((p = (nodeType*)malloc(sizeof(nodeType))) == NULL)
        printf("out of memory");

    if (!strcmp(s, "=")) {
        printf("wrong comparator\n");
        fflush(stdout);
        printf("wrong comparator");
    }
    /* copy information */
    p->type = typeAssgn;
    p->id.s = "=";
    p->id.iLength = strlen("=");
  //  printf("comp: %d %s \n", comparisonLength, s);
    return p;
}

nodeType *con(float value) {
    nodeType *p;

    /* allocate node */
    if ((p = (nodeType*)malloc(sizeof(nodeType))) == NULL)
        printf("out of memory");

    /* copy information */
    p->type = typeCon;
    p->con.fValue = value;

    return p;
}

nodeType *opr(int oper, int nops, ...) {
    va_list ap;
    nodeType *p;
    int i;

    /* allocate node */
    if ((p = (nodeType*)malloc(sizeof(nodeType))) == NULL)
        printf("out of memory");
    if ((p->opr.op = (nodeType**)malloc(nops * sizeof(nodeType))) == NULL)
        printf("out of memory");

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

/*
int yyerror(const char *s) {
    fprintf(stdout, "%s\n", s);
    // should this return 1? 
    return 1;
}
*/

