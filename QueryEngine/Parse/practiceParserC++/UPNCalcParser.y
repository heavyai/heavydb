%name UPNCalcParser
%define LSP_NEEDED
%define ERROR_BODY = 0
%define LEX_BODY = 0
%define LTYPE upn_ltype_t

%header{
#include <iostream>
#include <fstream>
%}

%union {
	int itype;
	char ctype;
}
%header{
typedef struct
{
	int first_line;
	int last_line;
	int first_column;
	int last_column;
	char *text;
} upn_ltype_t;
%}

%token UNKNOWN
%token <ctype> PLUS MINUS
%token <itype> NUMBER

%type <itype> number expression calculation

%start calculation

%%

calculation
: expression { $$ = $1; }
;

expression
: number	{ $$ = $1; }
| expression expression PLUS	{ $$ = $1 + $2; }
| expression expression MINUS	{ $$ = $1 - $2; }
;

number
: NUMBER	{ $$ = $1; }
;

%%
