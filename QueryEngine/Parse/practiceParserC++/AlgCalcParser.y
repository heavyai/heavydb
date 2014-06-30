%name AlgCalcParser
%define LSP_NEEDED
%define ERROR_BODY = 0
%define LEX_BODY = 0
%define LTYPE alg_ltype_t

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
} alg_ltype_t;
%}

%token UNKNOWN
%token <ctype> PLUS MINUS EQUALS
%token <itype> NUMBER

%type <itype> number subexpression
%type <itype> addexpression expression calculation

%start calculation

%%

calculation
: expression EQUALS	{ $$ = $1; }
;

expression
: addexpression	{ $$ = $1; }
;

addexpression
: subexpression	{ $$ = $1; }
| addexpression PLUS subexpression	{ $$ = $1 + $3; }
;

subexpression
: number	{ $$ = $1; }
| subexpression MINUS number	{ $$ = $1 - $3; }
; 

number
: NUMBER	{ $$ = $1; }
;

%%
