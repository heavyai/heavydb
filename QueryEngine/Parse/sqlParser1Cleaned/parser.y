%name CalcParser
%define LSP_NEEDED
%define ERROR_BODY = 0
%define LEX_BODY = 0
 
%header{
#include <iostream>
#include <fstream>
%}
 
%union {
	int i_type;
	char c_type;
}
 
%token UNKNOWN
%token <c_type> PLUS MINUS EQUALS
%token <i_type> NUMBER
 
%type <i_type> number subexpression addexpression
%type <i_type> expression calculation
 
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