name Scanner
define IOSTREAM
%{
using namespace std;
%}
DIGIT   [0-9]
DIGIT1  [1-9]
 
%%
 
"+"               { cout << "operator <" << yytext[0] << ">" << endl; }
"-"               { cout << "operator <" << yytext[0] << ">" << endl; }
"="               { cout << "operator <" << yytext[0] << ">" << endl; }
{DIGIT1}{DIGIT}*  { cout << "  number <" << yytext    << ">" << endl; }
.                 { cout << " UNKNOWN <" << yytext[0] << ">" << endl; }
 
%%
 
int main(int argc, char ** argv)
{
	/*
	Scanner scanner;
	scanner.yylex(); */
	return 0;
}