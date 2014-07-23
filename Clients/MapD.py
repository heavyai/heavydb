import socket
import select
from struct import *
from time import time

class MapD:
    def __init__(self, host = "", port=7777):
        #default values below
        self.isOutput = False
        self.port = port
        self.host = host 
        self.output = ""
        self.result = ""
        self.sock = 0
        self.delim = "-|;"
        self.data = ""
        self.colcount = 0
        self.rowcount = 0
        self.statusMsg = ""
        self.colInfo = []
        self.colData = []
        self.timer = False
        self.time = 0.0
        self.isConnected = False
        self.cursorRow = 0
        

    def __del__ (self):
        if self.sock != 0:
            self.sock.close()

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  
        self.sock.settimeout(1.5)
        try:
            self.sock.connect((self.host,self.port))
            self.sock.settimeout(None)
            self.isConnected = True
        except Exception as x:
            print x
            raise RuntimeError("Connection Failure")



    def close(self):
        if self.sock != 0:
            #print "Socket close"
            self.sock.shutdown(socket.SHUT_RDWR)
            self.sock.close()
            self.sock = 0
            self.isConnected = False

    def send(self, message = ""):
        message = message + "\n"
        msgLen = len(message)
        totalSent = 0
        while totalSent < msgLen:
            sent = self.sock.send(message[totalSent:])
            if sent == 0:
                raise RuntimeError("Connection broken")
            totalSent += sent
            


    def receive(self):
        #print "At receive"

        #if isBinary == False:
        #    self.result = ""
        #    delimPos = -1
        #    while True:
        #        chunk = self.sock.recv(8192)
        #        delimPos = chunk.find(self.delim)
        #        if delimPos >= 0:
        #            self.result += chunk[:delimPos]
        #            break
        #        else:
        #            self.result += chunk
        #else:

        totalLen = 0
        self.data = ""
        endOfStream = False
        isPackBegin = True
        while not endOfStream:
            pos = 0
            chunk = self.sock.recv(8192)
            #print chunk
            chunkSize = len(chunk)
            #for c in range(chunkSize):
                #print "Char at %d: %d" % (c, ord(chunk[c]))
            #print "Chunksize: %s" % chunkSize
            while True:
                #print "Pos: %d" % (pos,)
                if isPackBegin:
                    packLen = unpack ('I',chunk[pos:pos+4])[0]
                    pos += 4
                    totalLen += packLen
                    #print "PackLen: %d" % (packLen,)
                    if packLen == 0:
                        endOfStream = True
                        #print "eos"
                        break
                if packLen > chunkSize - pos:
                    self.data += chunk[pos:]
                    packLen -= chunkSize - pos
                    isPackBegin = False
                    break
                else:
                    self.data += chunk[pos:pos+packLen]
                    pos += packLen
                    isPackBegin = True
                    if pos == chunkSize: #if at end of chunk
                        break
                #raw_input("press enter")
        #print "Total len: %s" % totalLen
        #self.emptySocket()
        if totalLen > 0:
            #print "process data" 
            self.processData()
        self.cursorRow = 0
        
    def emptySocket(self):
        #print "emptying socket"
        input = [self.sock]
        while 1:
            inputready, o, e = select.select(input,[],[],0.0)
            if len(inputready)==0: break
            for s in inputready: s.recv(1)

    def bufferPos(self, pos): 
        bufferLen = pos % 4 
        if bufferLen > 0:
            bufferLen = 4 - bufferLen
        return pos + bufferLen

    def fetchone(self):
        if self.cursorRow < self.rowcount:
            returnList = [None]* self.colcount
            for c in range(self.colcount):
                returnList[c] = self.colData[c][self.cursorRow]
            self.cursorRow += 1
            return tuple(returnList)
        else:
            return None

    def fetchall(self):
        resultRows = []
        resultTup = self.fetchone()
        while resultTup != None:
            resultRows.append(resultTup)
            resultTup = self.fetchone()
        return resultRows


    
    def readStr (self, pos, col): 
        strLen = unpack('I',self.data[pos:pos+4])[0]
        formatStr = str(strLen) + 's'
        self.colData[col].append(unpack(formatStr, self.data[pos+4:pos+4+strLen])[0])
        pos += 4 + strLen
        pos = self.bufferPos(pos)
        return pos

    def readColTitle (self, pos): 
        strLen = unpack('I',self.data[pos:pos+4])[0]
        formatStr = str(strLen) + 's'
        title = unpack(formatStr, self.data[pos+4:pos+4+strLen])[0]
        pos += 4 + strLen
        pos = self.bufferPos(pos)
        return pos, title

    def processData(self):
        #print self.data
        self.isOutput=True
        self.rowcount = unpack('i', self.data[:4])[0]
        #print "Row count: %s" % self.rowcount
        pos = 4
        if self.rowcount < 0:
            pos, self.statusMsg = self.readColTitle(pos)
        else: 
            self.colcount = unpack('I', self.data[pos:12])[0]
            #meta = unpack ('iI',self.data[:8])
            #self.rowcount = meta[0]
            #self.colcount = meta[1]
            #print "NumTups: %d" % (self.rowcount,)
            #print "NumCols: %d" % (self.colcount,)
            self.colInfo = [] 
            self.colData = [] 
            pos = 12 
            for col in range(self.colcount):
                pos, name = self.readColTitle(pos) 
                pos, colType = self.readColTitle(pos) 
                byteLen = 1
                if colType == "bool":
                    frmt = "?"
                    byteLen = 1
                elif colType == "char":
                    frmt = "c"
                    byteLen = 1
                elif colType == "int":
                    frmt = "i"
                    byteLen = 4
                elif colType == "timestamp":
                    frmt = "i"
                    byteLen = 4
                elif colType == "unsigned int":
                    frmt = "I"
                    byteLen = 4
                elif colType == "unsigned long":
                    frmt = "Q"
                    byteLen = 8
                elif colType == "float":
                    frmt = "f"
                    byteLen = 4
                elif colType == "double":
                    frmt = "d"
                    byteLen = 8
                elif colType == "varchar":
                    frmt = "s"

                self.colInfo.append((name, colType, frmt, byteLen))
                self.colData.append([])

            #for c in self.colInfo:
            #    print c
            
            for tup in range(self.rowcount):
                for col in range(self.colcount):
                    frmt = self.colInfo[col][2]
                    if frmt != "s":
                        self.colData[col].append(unpack(frmt, self.data[pos:pos+self.colInfo[col][3]])[0])
                        pos += self.colInfo[col][3]
                    else:
                        pos = self.readStr(pos, col) 
                        
            #print self.colData

        

    def escapeQuery(self, message, params): 
        #print "escape query"
        numVars = message.count('%s') 
        if numVars != len(params):
            raise RuntimeError("Bad query")
        #message = message.replace("'", "''")
        formattedParams = []
        #print "hello"
        #print len(params)
        for param in params:
            var = param

            if var == None:
                var = "null"
            elif type(var) == str or type(var) == unicode:
                var = var.replace("'", "''")
                var = "'" + var + "'"
            elif type(var) == bool:
                if var == True:
                    var = "true"
                else:
                    var = "false"
            else:
                var = str(var)
            formattedParams.append(var)
        formattedParams = tuple(formattedParams)
        query = message % formattedParams
        return query 

    def execute(self, message = "", params = (), timeout = None ):
        if self.sock != 0:
            self.sock.settimeout(timeout)
            try:
                startTime = 0.0
                endTime = 0.0
                message = self.escapeQuery(message,params)
                #print message
                self.send(message)
                if self.timer == True:
                    startTime = time()
                self.isOutput = False
                self.receive()
                if self.timer == True:
                    endTime = time()
                    self.time = (endTime - startTime) * 1000.0 

            except Exception as x:
                self.sock.close()
                print x
                raise RuntimeError("Failure communicating over socket")
        else:
            raise RuntimeError("No open connection")
