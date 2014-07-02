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
#include "ast/Name.h"
#include "ast/Table.h"

// define stack element type to be a 
// pointer to an AST node
#define YY_Parser_STYPE ASTNode*

extern ASTNode* parse_root;

// Variables declared in scanner.l
extern std::string strData[10];

using namespace std;

%}

%token AS
%token DROP
%token NAME
%token TABLE

%token INTNUM
%token UNKNOWN

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
: DROP TABLE table 				{ $$ = new BaseTableDef("DROP", (Table*)$3); }
;

table
: NAME 							{ $$ = new Table(strData[0]); }
;

%%
