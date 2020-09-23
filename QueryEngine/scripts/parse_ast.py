from __future__ import print_function

import sys
import re

if len(sys.argv) > 1:
    with open(sys.argv[1], 'r') as f:
        for line in f:
            if 'FunctionDecl' in line and 'ExtensionFunctions' in line:
                line = re.sub("-FunctionDecl.*line:[0-9]+:[0-9]+", "", line)
                print(line)
else:
    for line in sys.stdin:
        if 'FunctionDecl' in line and 'ExtensionFunctions' in line:
            line = re.sub("-FunctionDecl.*line:[0-9]+:[0-9]+", "", line)
            print(line, end ='')
