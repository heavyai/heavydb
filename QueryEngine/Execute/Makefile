main: Codec.cpp Translator.cpp main.cpp Codec.h Translator.h runtime_ll.o
	clang++ -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS -rdynamic -std=c++11 -I`llvm-config --includedir` -Wall -O3 -o main runtime_ll.o main.cpp Codec.cpp Translator.cpp `llvm-config --ldflags --libs` -ldl -lglog -lpthread -ltinfo -lz

group_by_hash_test: GroupByHashTest.cpp RuntimeFunctions.cpp
	clang++ -std=c++11 -Wall -O3 -o group_by_hash_test $^ -lgtest -lpthread

RuntimeFunctions.ll: RuntimeFunctions.cpp
	clang++ -std=c++11 -O3 -c -S -emit-llvm RuntimeFunctions.cpp

runtime_ll.o: RuntimeFunctions.ll
	ld -r -b binary -o runtime_ll.o $^

clean:
	rm -f *.o
	rm -f RuntimeFunctions.ll
	rm -f main
	rm -f group_by_hash_test
