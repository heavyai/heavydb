## Steps for Adding or Updating System Dashboards
1. Startup server with the `--allow-system-dashboard-update` flag.
2. Create new (system) dashboard in Immerse, under the `information_schema` database, or update an existing dashboard.
3. Save and download dashboard.
4. Navigate to the `SystemDashboards` directory and execute the `add_sys_dashboard.sh` script. For example,
```
sh add_sys_dashboard.sh /path/to/downloaded/dashboard.json dashboard_file_name.json
```
5. Add a `"system_dashboard_version":"v{dashboard version}"` entry to the JSON string on the second line of the 
added/updated dashboard JSON file.
6. Add the new dashboard JSON file name to the `dashboard_json_files` array in the 
`Catalog::initializeSystemDashboards()` method and re-build the server (`make -j 4 heavydb`). This step can be skipped 
for existing dashboard updates, where the file name is already in the array.
7. Run `make system_dashboard_data`.
8. Restart server and ensure that dashboard updates have been applied as expected.
