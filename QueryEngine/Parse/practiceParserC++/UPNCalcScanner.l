%name UPNCalcScanner
%define IOSTREAM
%define LEX_PARAM YY_UPNCalcParser_STYPE *val, YY_UPNCalcParser_LTYPE *loc
%define MEMBERS public: int line, column;
%define CONSTRUCTOR_INIT : line(1), column(1)

%header{
#include <sstream>
#include "UPNCalcParser.h"
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
            return UPNCalcParser::PLUS;
          }
"-"       {
            ++column;
            val->ctype = yytext[0];
            return UPNCalcParser::MINUS;
          }
{D1}{D}*  {
            column += strlen(yytext);
			std::istringstream(yytext) >> val->itype;
            return UPNCalcParser::NUMBER;
          }
.         {
            ++column;
            val->ctype = yytext[0];
            return UPNCalcParser::UNKNOWN;
          }
<<EOF>>   {
            yyterminate();
          }

%%
