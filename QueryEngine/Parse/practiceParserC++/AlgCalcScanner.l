%name AlgCalcScanner
%define IOSTREAM
%define LEX_PARAM YY_AlgCalcParser_STYPE *val, YY_AlgCalcParser_LTYPE *loc
%define MEMBERS public: int line, column;
%define CONSTRUCTOR_INIT : line(1), column(1)

%header{
#include <sstream>
#include "AlgCalcParser.h"
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
