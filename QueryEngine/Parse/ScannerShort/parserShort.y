%name Parser
%define LSP_NEEDED
%define MEMBERS                 \
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
#include "visitor/XMLTranslator.h"
#include <FlexLexer.h>


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

    Program program;
    SQLList sqlList;
    SQL sql;
    Schema sch;
    BaseTableDef btd;
    Table tbl;

};

%token <sName> NAME

%token DROP TABLE

%type <Program> program
%type <SQLList> sql_list
%type <SQL> sql
%type <Schema> schema
%type <BaseTableDef> base_table_def
%type <Table> table

%%

program:
	sql_list	{ Program program1($1); (exit(0); }
	;

sql_list:
	sql ';'		{ SQLList sqlList1($1); }
	;

sql:			
	schema		{ SQL sql1($1); }
	;

schema:
	base_table_def	{ BaseTableDef btd1($1); }
	;

base_table_def:
	DROP TABLE table { Table tbl1($3); }

table:
	NAME			{ $$ = $1; }
	;
 %%