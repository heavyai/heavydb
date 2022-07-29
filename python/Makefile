
.PHONY: thrift
thrift:
	thrift -r -gen py ../heavy.thrift
	rm -rf heavydb/thrift/*
	cp -r ../gen-py/heavydb/thrift/ heavydb/thrift/

.PHONY: build
build: thrift
	flit build

.PHONY: publish
publish: build
	flit publish
