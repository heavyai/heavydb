%option c++
%option noyywrap

%{
#include<sstream>
#include "parser.h"
#include <cstdlib>
#include <iostream>
using namespace std;

//YY_CalcParser_STYPE val;
%}

DIGIT   [0-9]
DIGIT1  [1-9]

%%

{DIGIT1}{DIGIT}*  {
                  std::cout <<"Lexer: "<< yytext <<std::endl;
                  return Parser::NUMBER;
                 }
.                 {
                   return Parser::UNKNOWN;
                 }
<<EOF>>           {
                   yyterminate();
                 }

%%