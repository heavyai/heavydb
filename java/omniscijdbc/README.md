
JDBC Integration Testing Notes.

    1. Check out and build the branch to be tested. 
    2. Use initdb to create a fresh instance of the  default database.
    3. Start the OmniSci server to run over the new database instance.
    4. Start the OmniSci web server specifying  the --cert, --key and --enable-https options.  Test certificates can be found in the resources folder of the project.  
    5. There are a range of configuration files in the resources directory; these files expect the server to be running on 'localhost'.  To connect to a server running on a different host or port, these files will need to be edited.
    6. From the directory `<project dir>/java` run the command `mvn test -D -DskipTests=false -Domnisci.release.version=4.5.0`

Note:
    The version in the mvn command needs to match the version listed in the top level CMakeLists.txt file.  If this is a development build append "-SNAPSHOT" to the end of the version string
    If the connection string specified in the properties file is incorrect the tests may hang for an extended period of time
    Due to the method used for versioning in the pom.xml files, the mvn test command can not be run from the omniscijdbc directory
