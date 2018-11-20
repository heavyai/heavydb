"""
Run this script to create remote dictionary server.
"""
import remotedict
server = remotedict.Server().run(threaded=False, debug=True)
print('EOF', __name__)
