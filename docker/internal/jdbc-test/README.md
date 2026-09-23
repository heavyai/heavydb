## Docker setup for jdbc testing
The docker compose file in this directory starts two containers that can be used to facilitate testing of HEAVY.AI's jdbc driver. The file was primarily written to be used in the HEAVY.AI ci environment and its structure reflects this.

The first container runs a HEAVY.AI database based on a cpu only image that is updated nightly.

The second, contains the development stack required to build the HEAVY.AI product. Loading the jdbc source code into this container allows the associated maven tests to be run against the data base in the first container.

###Requirements 
Docker, Docker Compose and a vpn connection to the HEAVY.AI network.

###Steps

1. Edit the '.env' file in this this directory, setting BUILD_TMP_DIR to point to the source code's parent directory and MVNHOME to either the '.m2' directory in your home directory or an alternate '.m2' folder.

2. Edit the jdbc java test properties files to connect to 'heavydbserver' rather than 'localhost'.  The simple 'for' loop below run from the test resources folder can be used for this task.

`for i in *.properties ; do echo $i ; sed "s/localhost/heavydbserver/" $i > $i.bck; cp $i.bck $i; done`


3. Run `sudo docker-compose up -d --remove-orphans` from this directory (the directory containing the 'compose.yml' file).  Note the command 'sudo docker-compose config' can be used to verify the settings from the '.env' file are being correctly inserted into the yaml file.


3. Set the environment variable VER to the current jdbc release.  For example '6.1.0-SNAPSHOT'


4. Run `sudo docker-compose exec -T buildhost bash -c "cd /heavydb/java/heavyaijdbc/; mvn test -DskipTests=false -Domnisci.release.version=$VER -Dthrift.version=0.13.0 -Dtest=!ai.heavy.jdbc.HeavyAIConnectionTest#*_encrypted*+tst2_http_unencrypted+tst5_properties_connection"`


###Caveats 

Running `mvn test` from within the docker container will potentially create files on the shared drives owned by the root user; causing permissions problems in the future.

Using `docker-compose exec`allows direct use of the service name defined in the compose file rather than the decorated name used by docker.  Unfortunately at this time `docker-compose exec` was unable to interpret the --workdir options; for this reason the exec command includes a `cd` and is run as `bash`
###Possible Extensions 

By sharing server configuration files with the heavydbserver alternate communications protocols, such as https could also be tested.
