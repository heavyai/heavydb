from __future__ import print_function

import sys
import re

for line in sys.stdin:
    if 'FunctionDecl' in line and 'ExtensionFunctions' in line:
        line = re.sub("-FunctionDecl.*line:[0-9]+:[0-9]+", "", line)
        print(line, end ='')
