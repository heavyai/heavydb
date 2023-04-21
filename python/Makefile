
.PHONY: thrift
thrift:
	thrift -r -gen py ../heavy.thrift
	rm -rf heavydb/thrift/*
	cp -r gen-py/heavydb/thrift/* heavydb/thrift/
	cp -r gen-py/heavydb/common/* heavydb/common/
	cp -r gen-py/heavydb/completion_hints/* heavydb/completion_hints/
	cp -r gen-py/heavydb/extension_functions/* heavydb/extension_functions/
	cp -r gen-py/heavydb/serialized_result_set/* heavydb/serialized_result_set/

.PHONY: build
build: thrift
	flit build

.PHONY: publish
publish: build
	flit publish

.PHONY: clean
clean:
	rm -r heavydb/thrift/*