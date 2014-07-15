import re

def buildVisitFunction(nodeName):
	s = "\tvoid visit(class " + nodeName + " *v) {\n"
	s = s + "\t\tprintTabs(INCR);\n\t\tcout << \"<" + nodeName + ">\" << endl;\n\t\t"
	s = s + "\n\n"

	childNameList = []

	with open('childNames.txt', 'r') as nameFile:
		childNameData = nameFile.read()
		childNameList = re.split("\n", childNameData)
	
#	print(childNameList)
	for each in childNameList:
		s = s + "\t\tif (v->" + each + ")  v->" + each+ "->accept(*this);\n"

	s = s + "\n\t\tprintTabs(DECR);\n\t\tcout << \"</" + nodeName + ">\" << endl;\n\t}\n"

	return s;

nodeList = []
with open('allIncludes.txt', 'r') as f:

    readList = f.read()
    a = readList.split("\n")
    for each in a:
        b = re.split("[/.]", each)
        if len(b) > 1:
        	nodeList.append(b[1])


#print(nodeList)

for each in nodeList:
	
	print(buildVisitFunction(each))


