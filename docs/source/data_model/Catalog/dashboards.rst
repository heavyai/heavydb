``Dashboards`` are classes that hold the state and metadata of dashboards created in Immerse(Front-end visualization tool). The information is stored in the ``mapd_dashboards`` table and also in the ``DashboardDescriptorMap`` map.
Below is the structure of in-memory ``DashboardDescriptor`` object that holds dashboard information in-memory:

.. code-block:: cpp

    struct DashboardDescriptor {
        int32_t dashboardId;       /**< dashboardId starts at 0 for valid dashboard. */
        std::string dashboardName; /**< dashboardName is the name of the dashboard. dashboard
                                        -must be unique */
        std::string dashboardState;
        std::string imageHash;
        std::string updateTime;
        std::string dashboardMetadata;
        int32_t userId;
        std::string user;
        std::string dashboardSystemRoleName; /** Stores system role name */
    };