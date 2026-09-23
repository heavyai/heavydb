## Setup for ODBC FSI Testing Against Different Databases
This page contains instructions for setting up an environment for developing and testing against different
databases when working on the ODBC FSI data wrapper. The setup documented here leverages docker for easier 
database installation, configuration, and testing.

###Setup Steps
1. Install Docker and the NVIDIA Container Toolkit (see [parent documentation](../../README.md) for more details).

2. Install Docker Compose following the instructions specified [here](https://docs.docker.com/compose/install/).

3. Create a symbolic link `~/odbc-fsi-src` that points to your source directory. 
For example `ln -sfn /path/to/omniscidb-internal ~/odbc-fsi-src`.

4. Navigate to the `docker/internal/odbc-fsi-dev` directory, which contains the `docker-composes.yml` file.

5. Run `docker-compose build --pull` in order to apply any updates made to the `image-build/Dockerfile` file. 
Note that a VPN connection is required to pull the OmniSciDB docker image when running the command for the 
first time. 

6. Run `docker-compose up -d` to start up all docker containers for testing.

7. Connect to the development container by running `docker exec -it odbc-fsi-dev_odbc-fsi-dev_1 /bin/bash`.
This should take you to the source directory within the docker container where you can go through the
normal build process i.e. creating a build directory, running 
`cmake -DCMAKE_BUILD_TYPE=debug ..`, `make -j 4`, etc.

8. After a development/testing session, exit the development container and run `docker-compose down` to shutdown 
all test docker containers.

###Adding New Test Databases
As ODBC FSI development proceeds, there will be a need to add new databases for testing.
The following are instructions on how to do this:

1. Find the docker image for the latest version of the database to be added (this will typically be on 
[docker hub](https://hub.docker.com/)). If there are no docker images for the database, talk to the
dev-ops team about the possibility of setting up one locally.

2. Add a new service to the `docker-compose.yml` file along with any required configuration for the new 
database. The current convention is to have the service name match the database name and to use a host 
port number that is the previous service's host port number plus one.

3. Update the `Dockerfile` in the `image-build` subdirectory to include an installation of the ODBC driver for the 
new database and to add a symbolic link to the driver file from `/usr/lib/odbc-drivers/{driver file name}`.

4. Add a data source configuration for the new database to the `odbc.ini` file in the `image-build` subdirectory.
