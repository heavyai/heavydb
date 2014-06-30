
%option c++
%option noyywrap

%{
#include <sstream>
#include "AlgCalcParser.h"
#define IOSTREAM
#define LEX_PARAM YY_CalcParser_STYPE *val, YY_CalcParser_LTYPE *loc
#define MEMBERS public: int line, column;
#define CONSTRUCTOR_INIT : line(1), column(1)
%}

D   [0-9]
D1  [1-9]

%%

" "       {
            ++column;
          }
"\t"      {
            column += 8;
          }
"+"       {
            ++column;
            val->ctype = yytext[0];
            return AlgCalcParser::PLUS;
          }
"-"       {
            ++column;
            val->ctype = yytext[0];
            return AlgCalcParser::MINUS;
          }
"="       {
            ++column;
            val->ctype = yytext[0];
            return AlgCalcParser::EQUALS;
          }
{D1}{D}*  {
            column += strlen(yytext);
            std::istringstream(yytext) >> val->itype;
            return AlgCalcParser::NUMBER;
          }
.         {
            ++column;
            val->ctype = yytext[0];
            return AlgCalcParser::UNKNOWN;
          }
<<EOF>>   {
            yyterminate();
          }

%%
