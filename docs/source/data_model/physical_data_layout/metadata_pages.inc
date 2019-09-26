Each table directory also contains metadata files. These files by default have a `page_size` of **4096** and will appear on the file system as ``<file_number>.4096.mapd``. Each file contains **4096** pages and is therefore 16MBytes on disk (4096 bytes x 4096 pages).

They contain metadata information about each page in the data files.

The content of the metadata has a header much like a datafile but the *page_id* will be **-1** interpreted as the metadata page. A new metadata page is written out for a chunk each time the chunk contents are changes; the current metadata page for a chunk is the one with the highest valid epoch.

Metadata page store the same metadata records stored about chunks in the memory data layer.

Currently the database does not have a process to delete old metadata pages. This can cause delays at start up as the system reads old records.
