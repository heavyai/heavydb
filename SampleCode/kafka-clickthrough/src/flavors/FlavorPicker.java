package flavors;

// Swing/AWT Interface classes
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.EventQueue;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;

// Generic Java properties object
import java.util.Properties;

// Kafka Producer-specific classes
import org.apache.kafka.clients.producer.KafkaProducer;
import org.apache.kafka.clients.producer.Producer;
import org.apache.kafka.clients.producer.ProducerRecord;

public class FlavorPicker{

    private JFrame frmFlavors;
    private Producer<String, String> producer;
    /**
     * Launch the application.
     */
    public static void main(String[] args) {
       EventQueue.invokeLater(new Runnable() {
          public void run() {
             try {
                FlavorPicker window = new FlavorPicker(args);
                window.frmFlavors.setVisible(true);
             } catch (Exception e) {
                e.printStackTrace();
             }
          }
       });
    }

    /**
     * Create the application.
     */
    public FlavorPicker(String[] args) {
       initialize(args);
    }

    /**
     * Initialize the contents of the frame.
     */
    private void initialize(String[] args) {
       frmFlavors = new JFrame();
       frmFlavors.setTitle("Flavors");
       frmFlavors.setBounds(100, 100, 408, 177);
       frmFlavors.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
       frmFlavors.getContentPane().setLayout(null);
       
       final JLabel lbl_yourPick = new JLabel("You picked nothing.");
       lbl_yourPick.setBounds(130, 85, 171, 15);
       frmFlavors.getContentPane().add(lbl_yourPick);
       
       JButton button = new JButton("Strawberry");
       button.addActionListener(new ActionListener() {
          public void actionPerformed(ActionEvent arg0) {
             lbl_yourPick.setText("You picked strawberry.");
             pick(args,1);
          }
       });
       button.setBounds(141, 12, 114, 25);
       frmFlavors.getContentPane().add(button);
       
       JButton btnVanilla = new JButton("Vanilla");
       btnVanilla.addActionListener(new ActionListener() {
          public void actionPerformed(ActionEvent e) {
             lbl_yourPick.setText("You picked vanilla.");
             pick(args,2);
          }
       });
       btnVanilla.setBounds(278, 12, 82, 25);
       frmFlavors.getContentPane().add(btnVanilla);

       
       JButton btnChocolate = new JButton("Chocolate");
       btnChocolate.addActionListener(new ActionListener() {
          public void actionPerformed(ActionEvent e) {
             lbl_yourPick.setText("You picked chocolate.");
             pick(args, 0);
          }
       });

       btnChocolate.setBounds(12, 12, 105, 25);
       frmFlavors.getContentPane().add(btnChocolate);    
    }
    public void pick(String[] args,int x) {
         String topicName = args[0].toString();
         String[] value = {"chocolate","strawberry","vanilla"};
         
         // Set the producer configuration properties.
         Properties props = new Properties();
            props.put("bootstrap.servers", "localhost:9097");// 9097 to avoid Immerse:9092
            props.put("acks", "all");
            props.put("retries", 0);
            props.put("batch.size", 100);
            props.put("linger.ms", 1);
            props.put("buffer.memory", 33554432);
            props.put("key.serializer",
                "org.apache.kafka.common.serialization.StringSerializer");
            props.put("value.serializer",
                "org.apache.kafka.common.serialization.StringSerializer");

         // Instantiate a producerSampleJDBC
         producer = new KafkaProducer<String, String>(props);

         // Send a 1000 record stream to the Kafka Broker
         for (int y=0; y<1000; y++){
             producer.send(new ProducerRecord<String, String>(topicName, value[x]));
         }
    }
}//End FlavorPicker.java


