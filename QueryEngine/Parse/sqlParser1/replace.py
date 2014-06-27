
from __future__ import print_function
import re

with open('toReplace1', 'r') as f:
	data = f.read()
	b = re.split('\n', data)
	

	for each1 in b:
		a = re.split("%type <nPtr>", each1)
		for each2 in a:
			c = re.split("[ \t\n]", each2)
			for each3 in c:
				if each3 is not "":
					print("case " + each3.upper() + ':      s = \"' + each3.upper() + '\";       break;')
