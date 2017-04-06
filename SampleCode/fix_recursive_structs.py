#!/usr/bin/env python

from redbaron import RedBaron
import sys


def main():
    """Rewrite Thrift-generated Python clients to handle recursive structs. For
    more details see: https://issues.apache.org/jira/browse/THRIFT-2642.

    Requires package `RedBaron`, available via pip:
    $ pip install redbaron

    To use:

    $ thrift -gen py mapd.thrift
    $ mv gen-py/mapd/ttypes.py gen-py/mapd/ttypes-backup.py
    $ python fix_recursive_structs.py gen-py/mapd/ttypes-backup.py gen-py/mapd/ttypes.py

    """
    in_file = open(sys.argv[1], 'r')
    out_file = open(sys.argv[2], 'w')

    red_ast = RedBaron(in_file.read())

    thrift_specs = [ts.parent for ts in red_ast.find_all(
        'name', 'thrift_spec') if ts.parent.type == 'assignment' and ts.parent.parent.name in ['TDatumVal', 'TColumnData']]

    nodes = []
    for ts in thrift_specs:
        node = ts.copy()
        node.target = ts.parent.name + '.' + str(node.target)
        nodes.append(node)
        ts.value = 'None'

    red_ast.extend(nodes)
    out_file.write(red_ast.dumps())


if __name__ == '__main__':
    main()
