List of databases with it's small amount of metadata is stored in ``mapd_databases`` table. ``DBMetadata`` holds the in-memory details of the database. ``getMetadataForDB*`` method provides database details.
Below is the structure of ``DBMetadata``:

.. code-block:: cpp

    struct DBMetadata {
        DBMetadata() : dbId(0), dbOwner(0) {}
        int32_t dbId;
        std::string dbName;
        int32_t dbOwner;
    };