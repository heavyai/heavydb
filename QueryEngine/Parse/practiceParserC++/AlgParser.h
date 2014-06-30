#ifndef __ALGPARSER_H__
#define __ALGPARSER_H__

#include "AlgCalcScanner.h"
#include "AlgCalcParser.h"
#include "Parser.h"

class AlgParser : public AlgCalcParser, virtual public Parser
{
	private:
		AlgCalcScanner scanner;
	public:
		AlgParser()
			{}
		virtual int yylex();
		virtual void yyerror(char * msg);
		virtual int parse();
};

#endif
