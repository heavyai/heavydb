The user's data is stored in the ``mapd_users`` table. Instances of the C++ struct ``UserMetadata`` provides an in-memory representation of user stored data. User details can be accessing method ``getMetadataForUser*`` on the ``SysCatalog`` class.
Below is the structure of ``UserMetadata``:

.. code-block:: cpp

    struct UserMetadata {
        UserMetadata(int32_t u, const std::string& n, const std::string& p, bool s, int32_t d)
            : userId(u), userName(n), passwd_hash(p), isSuper(s), defaultDbId(d) {}
        UserMetadata() {}
        UserMetadata(UserMetadata const& user_meta)
            : UserMetadata(user_meta.userId,
                            user_meta.userName,
                            user_meta.passwd_hash,
                            user_meta.isSuper.load(),
                            user_meta.defaultDbId) {}
        int32_t userId;
        std::string userName;
        std::string passwd_hash;
        std::atomic<bool> isSuper;
        int32_t defaultDbId;
    };