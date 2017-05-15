# !/usr/bin/env python

import jaydebeapi


def connect(dbname, user, host, password):
    jar = './mapdjdbc-1.0-SNAPSHOT-jar-with-dependencies.jar'  # may want to parametrize
    try:
        return jaydebeapi.connect('com.mapd.jdbc.MapDDriver',
                                  'jdbc:mapd:{}:{}:'.format(host, dbname),
                                  {'user': user, 'password': password},
                                  jar)
    except Exception as e:
        print('Error: {}'.format(str(e)))
        raise e
