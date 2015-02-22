<?php

// Setup the path to the thrift library folder
$GLOBALS['THRIFT_ROOT'] = './Thrift';
 
// Load up all the thrift stuff
require_once $GLOBALS['THRIFT_ROOT'].'/Base/TBase.php';
require_once $GLOBALS['THRIFT_ROOT'].'/Exception/TException.php';
require_once $GLOBALS['THRIFT_ROOT'].'/Exception/TApplicationException.php';
require_once $GLOBALS['THRIFT_ROOT'].'/StringFunc/TStringFunc.php';
require_once $GLOBALS['THRIFT_ROOT'].'/StringFunc/Core.php';
require_once $GLOBALS['THRIFT_ROOT'].'/Factory/TStringFuncFactory.php';
require_once $GLOBALS['THRIFT_ROOT'].'/Protocol/TProtocol.php';
require_once $GLOBALS['THRIFT_ROOT'].'/Protocol/TBinaryProtocol.php';
require_once $GLOBALS['THRIFT_ROOT'].'/Transport/TTransport.php';
require_once $GLOBALS['THRIFT_ROOT'].'/Transport/TBufferedTransport.php';
require_once $GLOBALS['THRIFT_ROOT'].'/Transport/TSocket.php';
require_once $GLOBALS['THRIFT_ROOT'].'/Type/TMessageType.php';
require_once $GLOBALS['THRIFT_ROOT'].'/Type/TType.php';
 
require_once $GLOBALS['THRIFT_ROOT'].'/gen-php/mapd/Types.php';
require_once $GLOBALS['THRIFT_ROOT'].'/gen-php/mapd/MapD.php';

$socket = new Thrift\Transport\TSocket('localhost', '9090');
$transport = new Thrift\Transport\TBufferedTransport($socket);
$protocol = new Thrift\Protocol\TBinaryProtocol($transport);

// Create a calculator client
$client = new mapd\MapDClient($protocol);

// Open up the connection
$transport->open();

var_dump($client->getColumnTypes("test"));
var_dump($client->select("SELECT COUNT(*) FROM test;"));
