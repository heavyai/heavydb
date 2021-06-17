.. OmniSciDB Quickstart

###########################
Start and Load Sample Data
###########################

Starting the Server
===================

The `startomnisci` wrapper script may be used to start OmniSciDB in a testing environment. This script performs the following tasks:

* initializes the server `data` directory via ``initdb``, if required
* starts the OmniSciDB server, ``omnisci_server``
* offers to download and import a sample dataset using the `insert_sample_data` script if flag ``--sample-data`` is provided

Assuming you are in the `build` directory, and it is a subdirectory of the `omniscidb` repository, `startomnisci` may be run by:

.. code-block::shell

    ../startomnisci

Starting Manually
-----------------

It is assumed that the following commands are run from inside the `build` directory.

Initialize the `data` storage directory. This command only needs to be run once.

.. code-block:: shell

    mkdir data && ./bin/initdb data

Start the OmniSciDB server:

.. code-block:: shell

    ./bin/omnisci_server

You can now start using the database. The `omnisql` utility may be used to interact with the database from the command line:

.. code-block::shell

    ./bin/omnisql -p HyperInteractive

where `HyperInteractive` is the default password. The default user `admin` is assumed if not provided.

Working With Data
=================

Users can always insert a sample dataset by running the included `insert_sample_data` script:

.. code-block:: shell

    ../insert_sample_data

OmniSciDB also provides a variety of utilities for loading data into a table:

* `COPY FROM <https://www.omnisci.com/docs/latest/6_loading_data.html#copy-from>`_
* `SQLImporter <https://www.omnisci.com/docs/latest/6_loading_data.html#sqlimporter>`_
* `StreamInsert <https://www.omnisci.com/docs/latest/6_loading_data.html#streaminsert>`_
* `Importing AWS S3 Files <https://www.omnisci.com/docs/latest/6_loading_data.html#importing-aws-s3-files>`_
* `KafkaImporter <https://www.omnisci.com/docs/latest/6_loading_data.html#kafkaimporter>`_
* `StreamImporter <https://www.omnisci.com/docs/latest/6_loading_data.html#streamimporter>`_
* `HDFS with Sqoop <https://www.omnisci.com/docs/latest/6_loading_data.html#hdfs>`_
