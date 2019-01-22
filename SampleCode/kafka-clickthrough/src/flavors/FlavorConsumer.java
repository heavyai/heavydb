package flavors;

import java.util.Properties;
import java.util.Arrays;
import org.apache.kafka.clients.consumer.KafkaConsumer;
import org.apache.kafka.clients.consumer.ConsumerRecords;
import org.apache.kafka.clients.consumer.ConsumerRecord;

// JDBC
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.SQLException;

// Usage:\n\n FlavorConsumer <kafka-topic-name> <mapd-database-password>

public class FlavorConsumer {
  public static void main(String[] args) throws Exception {
    if (args.length < 2) {
      System.out.println(
              "Usage:\nFlavorConsumer <kafka-topic-name> <mapd-database-password>");
      return;
    }
    // Configure the Kafka Consumer
    String topicName = args[0].toString();
    Properties props = new Properties();

    props.put("bootstrap.servers", "localhost:9097"); // Use 9097 so as not
                                                      // to collide with
                                                      // MapD Immerse
    props.put("group.id", "test");
    props.put("enable.auto.commit", "true");
    props.put("auto.commit.interval.ms", "1000");
    props.put("session.timeout.ms", "30000");
    props.put("key.deserializer",
            "org.apache.kafka.common.serialization.StringDeserializer");
    props.put("value.deserializer",
            "org.apache.kafka.common.serialization.StringDeserializer");
    KafkaConsumer<String, String> consumer = new KafkaConsumer<String, String>(props);

    // Subscribe the Kafka Consumer to the topic.
    consumer.subscribe(Arrays.asList(topicName));

    // print the topic name
    System.out.println("Subscribed to topic " + topicName);

    String flavorValue = "";

    while (true) {
      ConsumerRecords<String, String> records = consumer.poll(1000);

      // Create connection and prepared statement objects
      Connection conn = null;
      PreparedStatement pstmt = null;

      try {
        // JDBC driver name and database URL
        final String JDBC_DRIVER = "com.mapd.jdbc.MapDDriver";
        final String DB_URL = "jdbc:mapd:localhost:6274:mapd";

        // Database credentials
        final String USER = "mapd";
        final String PASS = args[1].toString(); // name and pw in cleartext?

        // STEP 1: Register JDBC driver
        Class.forName(JDBC_DRIVER);

        // STEP 2: Open a connection
        conn = DriverManager.getConnection(DB_URL, USER, PASS);

        // STEP 3: Prepare a statement template
        pstmt = conn.prepareStatement("INSERT INTO flavors VALUES (?)");

        // STEP 4: Populate the prepared statement batch
        for (ConsumerRecord<String, String> record : records) {
          flavorValue = record.value();
          pstmt.setString(1, flavorValue);
          pstmt.addBatch();
        }

        // STEP 5: Execute the batch statement (send records to MapD
        // Core Database)
        pstmt.executeBatch();

        // Commit and close the connection.
        conn.commit();
        conn.close();

      } catch (SQLException se) {
        // Handle errors for JDBC
        se.printStackTrace();

      } catch (Exception e) {
        // Handle errors for Class.forName
        e.printStackTrace();
      } finally {
        try {
          if (pstmt != null) {
            pstmt.close();
          }
        } catch (SQLException se2) {
        } // nothing we can do

        try {
          if (conn != null) {
            conn.close();
          }
        } catch (SQLException se) {
          se.printStackTrace();
        } // end finally try

      } // end try
    } // end main
  }
} // end FlavorConsumer}