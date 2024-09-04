
.PHONY: thrift
thrift:
	rm -rf heavydb/thrift/
	mkdir -p heavydb/thrift/
	mkdir -p heavydb/common/
	mkdir -p heavydb/completion_hints/
	mkdir -p heavydb/extension_functions/
	mkdir -p heavydb/serialized_result_set/
	# The thrift python generator builds __init__.py file(s).
	# If the generator is run in the python source directory
	# which contains __init__.py files, they will be over written,
	# To prevent this the make file uses the gen-py folder and
	# then cp the needed files in the directories that the python
	# source code's imports and package commands expect them in.
	#
	# The copied versions of the files are listed in the
	# .gitignore file in this dir and as generated files 
	# shouldn't be commited.
	# 
	thrift -r -gen py ../heavy.thrift
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
	rm -rf gen-py
	rm -rf heavydb/thrift/
	rm -rf heavydb/common
	rm -rf heavydb/completion_hints
	rm -rf heavydb/extension_functions
