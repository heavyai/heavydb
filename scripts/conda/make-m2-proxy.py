#!/usr/bin/env python

'''
Generate ~/.m2/settings.xml specifying proxies
if such environment variables are set.
'''

import os
import sys
import re
import errno

def simple_xml(name, sections):
    ''' very simple xml generator for one-level depth items '''
    result = ['<%s>' % name]
    for sec_name, sec_value in sections:
        result.append('  <{0}>{1}</{0}>'.format(sec_name, sec_value))
    result.append('</%s>' % name)
    return '\n'.join('    ' + line for line in result)

_made_ids = set()

def gen_proxy(var_name):
    value = os.environ.get(var_name, '')
    if not value:
        return None
    try:
        parsed = re.search(r'''((?P<protocol>[^:]+)://)?    # protocol followed by ://, optional
            ((?P<username>[^:]+)(:(?P<password>[^@]+))?@)?  # user:password part, optional
            (?P<host>[^@]+?)                                # hostname, which is basically everything but other known parts
            (:(?P<port>\d+))?                               # port, optional
            $''', value, re.VERBOSE).groupdict()
    except AttributeError:
        sys.stderr.write('WARNING: unexpected format, could not parse $%s=%s\n' % (var_name, value))
        return None

    if not parsed['host']:
        return None
    id_name = var_name.lower()
    if id_name in _made_ids:
        num = 0
        while ('%s.%s' % (id_name, num)) in _made_ids:
            num +=1
        id_name = '%s.%s' % (id_name, num)
    _made_ids.add(id_name)
    sections = [('id', id_name), ('active', 'true')]
    for param_name in ('protocol', 'host', 'port', 'username', 'password'):
        if parsed[param_name]:
            sections.append((param_name, parsed[param_name]))
    return simple_xml('proxy', sections)

def make_settings(*var_names):
    sections = []
    for name in var_names:
        value = gen_proxy(name)
        if value:
            sections.append(value)
    if not sections:
        return None
    template = '''<settings xmlns="http://maven.apache.org/SETTINGS/1.0.0"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://maven.apache.org/SETTINGS/1.0.0
                      https://maven.apache.org/xsd/settings-1.0.0.xsd">
  <proxies>
%s
  </proxies>
</settings>'''
    return template % '\n'.join(sections)

def main():
    settings = make_settings('http_proxy', 'HTTP_PROXY', 'https_proxy', 'HTTPS_PROXY')
    target = os.path.expanduser('~/.m2/settings.xml')
    if not settings:
        try:
            os.remove(target)
        except OSError as ex:
            if ex.errno != errno.ENOENT:
                raise
        return
    try:
        os.makedirs(os.path.dirname(target))
    except OSError as ex:
        if ex.errno != errno.EEXIST:
            raise
    with open(target, 'w') as out:
        out.write(settings)

if __name__ == '__main__':
    main()
