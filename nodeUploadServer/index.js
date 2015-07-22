var express = require('express');
var bodyParser = require('body-parser');
var multer  = require('multer');
var fs = require('fs');
var cors = require('cors');
var port = process.env.PORT || 8000;
var app = express();

var upload = multer({dest:'./uploads'});
var binaryUpload = upload.array('binary');

app.use(cors());

app.post('/upload', function (req, res) {
  binaryUpload(req, res, function (err) {
    if (err) {
      res.send('error');
    }
    console.log(req.files);
    res.send(200);
  })
});

var server = app.listen(port);