// calc.cc

#include <iostream>
using namespace std;

#include "AlgParser.h"
#include "UPNParser.h"

// usage:
//   calc a  : algebraic (default)
//   calc u  : UPN

int main(int argc, char ** argv)
{
	Parser * parser = 0;

	if (argc != 2) parser = new AlgParser;
	else switch (argv[1][0])
	{
		case 'u': parser = new UPNParser; break;
		default:
		case 'a': parser = new AlgParser; break;
	}

	if (parser->parse()) cerr << "Aborted" << endl;

	if (parser) delete parser;
	return 0;
}
