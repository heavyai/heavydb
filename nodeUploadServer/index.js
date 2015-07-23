var express = require('express');
var multer  = require('multer');
var fs = require('fs');
var cors = require('cors');
var port = process.env.PORT || 8000;
var app = express();

app.use(cors());

var mkdirSync = function (path) {
  try {
    fs.mkdirSync(path);
  } catch(e) {
    if ( e.code != 'EEXIST' ) throw e;
  }
}

mkdirSync('./uploads/');

var storage = multer.diskStorage({
  destination: function (req, file, cb) {
    var sessionId = req.headers.sessionid;
    mkdirSync('./uploads/' + sessionId);
    cb(null, './uploads/' + sessionId);
  },
  filename: function (req, file, cb) {
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
    res.send(200);
  })
});

var server = app.listen(port);

