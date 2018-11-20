"""
This module is imported by embedded Python in mapd_server on startup.
"""
import math
import remotedict

class Storages:
    """Storage interface for UDF function callers.
    """
    def __init__(self):
        self.cache = dict()    
    def __getitem__(self, name):
        storage = self.cache.get(name)
        if storage is None:
            self.cache[name] = storage = remotedict.Storage(name=name)
        return storage
storages = Storages()

