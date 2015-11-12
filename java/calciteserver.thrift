namespace java com.mapd.thrift.calciteserver


exception InvalidParseRequest {
  1: i32 whatUp,
  2: string whyUp
}

service CalciteServer {

   void ping(),

   string process(1:string user 2:string passwd 3:string catalog 4:string sqlText) throws (1:InvalidParseRequest parseErr)

}