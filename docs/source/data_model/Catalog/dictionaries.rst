Additional details of string dictionary columns are stored in ``mapd_dictionaries`` table. ``nbits`` denotes the compression parameter (32 bits/64 bits) of the string column.

Below is the structure of in-memory ``DictDescriptor`` object that holds dictionaries information in-memory:

.. code-block:: cpp

    struct DictDescriptor {
        DictRef dictRef;
        std::string dictName;
        int dictNBits;
        bool dictIsShared;
        std::string dictFolderPath;
        int refcount;
        bool dictIsTemp;
        std::shared_ptr<StringDictionary> stringDict;
        DictDescriptor(DictRef dict_ref,
                        const std::string& name,
                        int nbits,
                        bool shared,
                        const int rc,
                        std::string& fname,
                        bool temp)
            : dictRef(dict_ref)
            , dictName(name)
            , dictNBits(nbits)
            , dictIsShared(shared)
            , dictFolderPath(fname)
            , refcount(rc)
            , dictIsTemp(temp)
            , stringDict(nullptr) {}

        DictDescriptor(int db_id,
                        int dict_id,
                        const std::string& name,
                        int nbits,
                        bool shared,
                        const int rc,
                        std::string& fname,
                        bool temp)
            : dictName(name)
            , dictNBits(nbits)
            , dictIsShared(shared)
            , dictFolderPath(fname)
            , refcount(rc)
            , dictIsTemp(temp)
            , stringDict(nullptr) {
            dictRef.dbId = db_id;
            dictRef.dictId = dict_id;
        }
    };
