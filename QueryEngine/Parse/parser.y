%name Parser
%define LSP_NEEDED
%define MEMBERS                 \
    virtual ~Parser()   {} \
    private:                   \
       yyFlexLexer lexer;
%define LEX_BODY {return lexer.yylex();}
%define ERROR_BODY {cerr << "error encountered at line: "<<lexer.lineno()<<" last word parsed:"<<lexer.YYText()<<"\n";}

%header{
#include <iostream>
#include <fstream>
#include <FlexLexer.h>
#include <cstdlib>

// AST nodes
#include "ast/ASTNode.h"
#include "ast/Program.h"
#include "ast/SQLList.h"
#include "ast/SQL.h"
#include "ast/Schema.h"
#include "ast/BaseTableDef.h"
#include "ast/Table.h"
#include "ast/ColumnDef.h"
#include "ast/BaseTableElementCommalist.h"
#include "ast/BaseTableElement.h"
#include "ast/ColumnDefOpt.h"
#include "ast/ColumnDefOptList.h"
#include "ast/Literal.h"
#include "ast/DataType.h"
#include "ast/Column.h"

// define stack element type to be a 
// pointer to an AST node
#define YY_Parser_STYPE ASTNode*

extern ASTNode* parse_root;

// Variables declared in scanner.l
extern std::string strData[10];
extern int intData;

using namespace std;

%}

%token AS
%token DROP
%token NAME
%token TABLE
%token CREATE

%token INTNUM STRING
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

%token DOTNAME
%start program

%%

program
: sql_list						{ $$ = new Program((SQLList*)$1); parse_root = $$; }
|								{ $$ = 0; parse_root = $$; }
;

sql_list
: sql ';'						{ $$ = new SQLList((SQL*)$1); }
| sql_list sql ';'				{ $$ = new SQLList((SQLList*)$1, (SQL*)$2); }
;

sql
: schema 						{ $$ = new SQL((Schema*)$1); }
;

schema 							
: base_table_def				{ $$ = new Schema((BaseTableDef*)$1); }
;

base_table_def
: CREATE TABLE table '(' base_table_element_commalist ')'		{ $$ = new BaseTableDef("CREATE", (Table*)$3, (BaseTableElementCommalist*)$5); }
| DROP TABLE table 												{ $$ = new BaseTableDef("DROP", (Table*)$3); }
;

base_table_element_commalist
: base_table_element                                      { $$ = new BaseTableElementCommalist( (BaseTableElement*)$1); }
| base_table_element_commalist ',' base_table_element     { $$ = new BaseTableElementCommalist( (BaseTableElementCommalist*)$1, (BaseTableElement*)$3); }
;

base_table_element
:    column_def              { $$ = new BaseTableElement( (ColumnDef*)$1); }
// |   table_constraint_def    { $$ = new BaseTableElement( (TableConstraintDef*)$1); }
;

column_def
: column data_type column_def_opt_list        { $$ = new ColumnDef( (Column*)$1, (DataType*)$2, (ColumnDefOptList*)$3); }
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
   // |   CHECK '(' search_condition ')'              { $$ = new ColumnDefOpt(6, (SearchCondition*)$3); }
    |   REFERENCES table                            { $$ = new ColumnDefOpt(7, (Table*)$2); }
    //|   REFERENCES table '(' column_commalist ')'   { $$ = new ColumnDefOpt(8, (Table*)$2, (ColumnCommalist*)$4); }
    ;


literal
: NAME /* should be: STRING */           { $$ = new Literal(strData[0]); }
// | INTNUM                              { $$ = opr(LITERAL, 1, con($1)); }
// | APPROXNUM                           { $$ = opr(LITERAL, 1, con($1)); }
;

table
: NAME 							{ $$ = new Table(strData[0]); }
// | NAME '.' NAME     { $$ = opr(TABLE, 3, id($1), opr('.', 0), id2($3));}
// | NAME AS NAME      { $$ = opr(TABLE, 3, id($1), opr(AS, 0), id2($3));  }
;


/* data types */
data_type
: CHARACTER                           { $$ = new DataType(0); }
| CHARACTER '(' INTNUM ')'            { $$ = new DataType(0, intData); }
| VARCHAR                             { $$ = new DataType(1); }
| VARCHAR '(' INTNUM ')'              { $$ = new DataType(1, intData); }
| NUMERIC                             { $$ = new DataType(2); }
| NUMERIC '(' INTNUM ')'              { $$ = new DataType(2, intData); }
// |   NUMERIC '(' INTNUM ',' INTNUM ')'   { $$ = new DataType(2, strData[0], strData2[0]); }
| DECIMAL                             { $$ = new DataType(3); }
| DECIMAL '(' INTNUM ')'              { $$ = new DataType(3, intData); }
//   |   DECIMAL '(' INTNUM ',' INTNUM ')'   { $$ = new DataType(3, strData[0], strData2[0]); }
| INTEGER                             { $$ = new DataType(4); }
| SMALLINT                            { $$ = new DataType(5); }
| FLOAT                               { $$ = new DataType(6); }
| FLOAT '(' INTNUM ')'                { $$ = new DataType(6, intData); }
| REAL                                { $$ = new DataType(7); }
| DOUBLE PRECISION                    { $$ = new DataType(8); }
;


column
: NAME            { $$ = new Column(strData[0]); }
;

%%
