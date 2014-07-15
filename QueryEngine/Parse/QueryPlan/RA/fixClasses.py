import sys
import re

namespaceType = "RA"
fileName = sys.argv[1]
#print (sys.argv)
read_data = ""
with open(fileName, 'r') as inputf:
	read_data = inputf.read()

with open(fileName, 'w') as outputf:
	if not "namespace" in read_data:
		a = re.split("class|};", read_data)
		outputf.write(a[0])
		outputf.write("namespace " + namespaceType + "_Namespace {\n\tclass ")
		outputf.write(a[1])
		outputf.write("\t};\n}")
		outputf.write(a[2])

	else: 
		re.sub("//#include \"../visitor/Visitor(Old)?.h\"", "#include \"../visitor/Visitor.h\"", read_data) 
		#a = re.split("//#include \"../visitor/Visitor(Old)?\"", read_data)
#		print(read_data)
		outputf.write(a[0])
		outputf.write("#")
		outputf.write(a[1])