var express = require('express');
var multer  = require('multer');
var fs = require('fs');
var cors = require('cors');
var bodyParser = require('body-parser');
var port = process.env.PORT || 8000;
var app = express();
var path = require('path');

app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());
app.use(cors());


var parentDirectory = path.dirname(require.main.filename).split('/');
parentDirectory.pop();
parentDirectory = parentDirectory.join('/');
// serve up static assets
app.use(express.static(parentDirectory +  '/build/frontend/'));

var mkdirSync = function (path) {
  try {
    fs.mkdirSync(path);
  } catch(e) {
    if ( e.code != 'EEXIST' ) throw e;
  }
}

var deleteFolderRecursive = function(path) {
  if( fs.existsSync(path) ) {
    fs.readdirSync(path).forEach(function(file,index){
      var curPath = path + "/" + file;
      if(fs.lstatSync(curPath).isDirectory()) { 
        deleteFolderRecursive(curPath);
      } else { // delete file
        fs.unlinkSync(curPath);
      }
    });
    fs.rmdirSync(path);
  }
};

mkdirSync('./uploads/');

var storage = multer.diskStorage({
  destination: function (req, file, cb) {
    var sessionId = req.headers.sessionid;
    mkdirSync('./uploads/' + sessionId);
    req.toSendClient = path.dirname(require.main.filename) + '/uploads/' + sessionId
    cb(null, './uploads/' + sessionId);
  },
  filename: function (req, file, cb) {
    req.toSendClient += '/' + file.originalname
    cb(null, file.originalname);
  }
})

var upload = multer({ storage: storage })
var binaryUpload = upload.array('binary');

app.post('/upload', function (req, res) {
  binaryUpload(req, res, function (err) {
    if (err) {
      res.send('error');
    }
    res.send(req.toSendClient);
  })
});

app.post('/deleteUpload', function (req, res) {
  var folderPath = path.dirname(req.body.file);
  deleteFolderRecursive(folderPath);
  res.sendStatus(200);
});

var server = app.listen(port);