import com.mapd.thrift.server.MapD;
import com.mapd.thrift.server.TDatumType;
import com.mapd.thrift.server.TQueryResult;
import org.apache.thrift.TException;
import org.apache.thrift.protocol.TBinaryProtocol;
import org.apache.thrift.protocol.TJSONProtocol;
import org.apache.thrift.transport.THttpClient;
import org.apache.thrift.transport.TSocket;
import org.apache.thrift.transport.TSSLTransportFactory;
import org.apache.thrift.transport.TSSLTransportFactory.TSSLTransportParameters;
import org.apache.thrift.transport.TTransport;
import static java.lang.System.out;
import java.util.Date;

/*
Contact support@mapd.com with any questions

Install Thrift and generate MapD Thrift client
thrift -gen java mapd.thrift

Dependencies:
/thrift-0.9.3/lib/java/src/
/slf4j-api-1.7.21.jar
/slf4j-simple-1.7.21.jar
/thrift/gen-java/
/httpcore-4.2.3.jar
/httpclient-4.2.3.jar

Compile statement:
javac -cp /path/to/thrift-0.9.3/lib/java/src:/path/to/slf4j-api-1.7.21.jar:/path/to/thrift/gen-java/:/path/to/httpcore-4.2.3.jar:/path/to/httpclient-4.2.3.jar:. MapDExample.java

Execution example:
java -cp /path/to/thrift-0.9.3/lib/java/src:/path/to/slf4j-api-1.7.21.jar:/path/to/gen-java/:/path/to/httpcore-4.2.3.jar:/path/to/httpclient-4.2.3.jar:. MapDExample

Connection samples:
HTTP client - get_client('http://test.mapd.com:9091', null, true)
Binary protocol - get_client('locahost', 9091, false)

*/

public class MapDExample {
  public static void main (String[] args){
    String db_name = "mapd";
    String user_name = "mapd";
    String passwd = "HyperInteractive";
    String hostname = "http://test.mapd.com:9091";
    int portno = 9091;
    MapD.Client client;
    int session;
    String query;
    TQueryResult results;
    int numRows, numCols;
    boolean fieldIsArray;
    TDatumType fieldType;
    String fieldName, fieldType2;

    try{
      client = get_client(hostname, portno, false);
      session = client.connect(user_name, passwd, db_name);
      System.out.println("Connection complete");
      query = "select a,b from table1 limit 25;";
      System.out.println("Query is: " + query);

      // always use True for is columnar
      results = client.sql_execute(session, query, true, null);

      if (results.row_set.is_columnar) {
        numRows = results.row_set.columns.get(0).nulls.size();
        numCols = results.row_set.row_desc.size();
        for (int r = 0; r < numRows; r++) {
          for (int c = 0; c < numCols; c++) {
            fieldName = results.row_set.row_desc.get(c).col_name;
            fieldType = results.row_set.row_desc.get(c).col_type.type;
            fieldType2 = fieldType.toString();
            fieldIsArray = results.row_set.row_desc.get(c).col_type.is_array;
            System.out.println(fieldName);
            if (fieldIsArray) {
              System.out.println(results.row_set.columns.get(c).data.arr_col.get(r).data.str_col);
            }
            else {
              switch (fieldType2) {
                case "BOOL":
                  break;
                case "SMALLINT":
                case "INT":
                case "BIGINT":
                  System.out.println(results.row_set.columns.get(c).data.int_col.get(r));
                  break;
                case "FLOAT":
                case "DOUBLE":
                case "DECIMAL":
                  System.out.println(results.row_set.columns.get(c).data.real_col.get(r));
                  break;
                case "STR":
                  System.out.println(results.row_set.columns.get(c).data.str_col.get(r));
                  break;
                case "TIME":
                case "TIMESTAMP":
                case "DATE":
                  System.out.println(new Date(results.row_set.columns.get(c).data.int_col.get(r)*1000));
                  break;
                default:
                  break;
              }
            }
          }
        }

      } else {
        System.out.println("Please use columns not row in query execution");
        client.disconnect(session);
        System.exit(0);
      }
      client.disconnect(session);
    } catch (Exception x){
      x.printStackTrace();
    }

  }

  public static MapD.Client get_client(String host_or_uri, int port, boolean http) {
    THttpClient httpTransport;
    TTransport transport;
    TBinaryProtocol protocol;
    TJSONProtocol jsonProtocol;
    TSocket socket;
    MapD.Client client;

    try{
      if (http) {
         httpTransport = new THttpClient(host_or_uri);
         jsonProtocol = new TJSONProtocol(httpTransport);
         client = new MapD.Client(jsonProtocol);
         httpTransport.open();
         return client;
      }
      else {
         transport = new TSocket(host_or_uri, port);
         protocol = new TBinaryProtocol(transport);
         client = new MapD.Client(protocol);
         transport.open();
         return client;
       }
    } catch (TException x){
      x.printStackTrace();
    }
    return null;
  }

}
