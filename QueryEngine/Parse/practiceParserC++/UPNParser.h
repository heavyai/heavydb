#ifndef __UPNPARSER_H__
#define __UPNPARSER_H__

#include "UPNCalcScanner.h"
#include "UPNCalcParser.h"
#include "Parser.h"

class UPNParser : public UPNCalcParser, virtual public Parser
{
	private:
		UPNCalcScanner scanner;
	public:
		UPNParser()
			{}
		virtual int yylex();
		virtual void yyerror(char * msg);
		virtual int parse();
};

#endif
