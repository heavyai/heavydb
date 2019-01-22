using System;
using Thrift;
using Thrift.Protocol;
using Thrift.Server;
using Thrift.Transport;
using static MapD;
using static TDatumType;
using static TQueryResult;

/*

Contact support@mapd.com with any questions

Install Thrift and generate MapD Thrift client
thrift -gen csharp mapd.thrift

Dependencies:
/thrift/gen-csharp
ApacheThrift 0.10.0

Connection samples:
HTTP client - get_client('http://test.mapd.com:6274', null, true)
Binary protocol - get_client('locahost', 6274, false)

*/



namespace MapDExample
{
    public class MapDExample
    {
        public static void Main()
        {
            string db_name = "mapd";
			string user_name = "mapd";
			string passwd = "HyperInteractive";
			string hostname = "http://test.mapd.com";
            int portno = 6274;
            MapD.Client client;
            int session;
            string query;
            TQueryResult results;
            int numRows, numCols;
            bool fieldIsArray;
            TDatumType fieldType;
            string fieldName, fieldType2;
            try
            {
				client = get_client(hostname, portno, true);
				session = client.connect(user_name, passwd, db_name);
				Console.WriteLine("Connection Completed");
				query = "select a, b from table1 limit 10;";
				Console.WriteLine("Query is: " + query);
				results = client.sql_execute(session, query, true, null, -1, -1);

				if (results.Row_set.Is_columnar) {
					numRows = results.Row_set.Columns[0].Nulls.Count;
					numCols = results.Row_set.Row_desc.Count;
					for (int r = 0; r < numRows; r++) {
						for (int c = 0; c < numCols; c++) {
							fieldName = results.Row_set.Row_desc[c].Col_name;
							fieldType = results.Row_set.Row_desc[c].Col_type.Type;
							fieldType2 = fieldType.ToString();
							fieldIsArray = results.Row_set.Row_desc[c].Col_type.Is_array;
							Console.WriteLine(fieldName);
							if (fieldIsArray) {
								Console.WriteLine(results.Row_set.Columns[c].Data.Arr_col[r].Data.Str_col);
							}
							else {
								switch (fieldType2) {
								case "BOOL":
									break;
								case "SMALLINT":
								case "INT":
								case "BIGINT":
									Console.WriteLine(results.Row_set.Columns[c].Data.Int_col[r]);
									break;
								case "FLOAT":
								case "DOUBLE":
								case "DECIMAL":
									Console.WriteLine(results.Row_set.Columns[c].Data.Real_col[r]);
									break;
								case "STR":
									Console.WriteLine(results.Row_set.Columns[c].Data.Str_col[r]);
									break;
								case "TIME":
								case "TIMESTAMP":
								case "DATE":
									Console.WriteLine(new DateTime(1970, 1, 1, 0, 0, 0).AddMilliseconds(Convert.ToDouble(results.Row_set.Columns[c].Data.Int_col[r]*1000)));
									break;
								default:
									break;
								}
							}
						}
					}
				}
            }
            catch (TApplicationException x)
            {
                Console.WriteLine(x.StackTrace);
            }
        }

		public static MapD.Client get_client(string host_or_uri, int port, bool http) {
			THttpClient httpTransport;
			TTransport transport;
			TBinaryProtocol protocol;
			TJSONProtocol jsonProtocol;
			MapD.Client client;

			try{
				if (http) {
					Uri httpuri = new Uri(host_or_uri);
					httpTransport = new THttpClient(httpuri);
					jsonProtocol = new TJSONProtocol(httpTransport);
					client = new MapD.Client(jsonProtocol);
					httpTransport.Open();
					return client;
				}
				else {
					transport = new TSocket(host_or_uri, port);
					protocol = new TBinaryProtocol(transport);
					client = new MapD.Client(protocol);
					transport.Open();
					return client;
				}
			} catch (TException x){
				Console.WriteLine(x.StackTrace);
			}
			return null;
		}
    }
}
