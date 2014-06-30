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
#include <stdlib.h>
#include <FlexLexer.h>
using namespace std;
%}

%union {
       int i_type;
}

%token UNKNOWN
%token <i_type> NUMBER

%type <i_type> number

%start number

%%
number
: NUMBER { $$ = atoi(lexer.YYText());std::cout << "Parser value "<<$$<<std::endl;}
;

%%