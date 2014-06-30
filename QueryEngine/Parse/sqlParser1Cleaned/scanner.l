%name CalcScanner
%define IOSTREAM
%define LEX_PARAM YY_CalcParser_STYPE *val, YY_CalcParser_LTYPE *loc
%define MEMBERS public: int line, column;
%define CONSTRUCTOR_INIT : line(1), column(1)
 
%header{
#include<sstream>
#include "parser.h"
%}
 
DIGIT   [0-9]
DIGIT1  [1-9]
 
%%
 
" "               {
                    ++column;
                  }
"\t"              {
                    column += 8;
                  }
"+"               {
                    ++column;
                    val->c_type = yytext[0];
                    return CalcParser::PLUS;
                  }
"-"               {
                    ++column;
                    val->c_type = yytext[0];
                    return CalcParser::MINUS;
                  }
"="               {
                    ++column;
                    val->c_type = yytext[0];
                    return CalcParser::EQUALS;
                  }
{DIGIT1}{DIGIT}*  {
                    column += strlen(yytext);
                    std::istringstream(yytext) >> val->i_type;
                    return CalcParser::NUMBER;
                  }
.                 {
                    ++column;
                    val->c_type = yytext[0];
                    return CalcParser::UNKNOWN;
                  }
<<EOF>>           {
                    yyterminate();
                  }
 
%%