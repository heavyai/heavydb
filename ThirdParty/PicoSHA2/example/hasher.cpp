#include <iostream>
#include "../picosha2.h"

void CalcAndOutput(const std::string& src){
	std::cout << "src : \"" << src << "\"\n";
	std::cout << "hash: " << picosha2::hash256_hex_string(src) << "\n" << std::endl;
}

int main(int argc, char* argv[])
{
	if(argc == 1){
		CalcAndOutput("");
	}
	else{
		for(int i = 1; i < argc; ++i){
			CalcAndOutput(argv[i]);
		}
	}

    return 0;
}

