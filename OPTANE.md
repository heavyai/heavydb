Optane DCPMM Enabling
==============================

# Building

- To enable Optane DCPMM AppDirect (both volatile and persistent modes), add -ENABLE_DCPMM=on option to your cmake build. For example:

	cmake -DCMAKE_BUILD_TYPE=release -DENABLE_CUDA=off -DENABLE_DCPMM=on ..

	make -j96

# Running OmniSciDB with Optane DCPMM

- To run OmniSciDB with Optane DCPMM enabled, option --pmm or --pmm-store or both have to be specified when OmniSciDB is started. For example:

	./startomnisci --pmm ./vpmem.conf

	or

	./startomnisci --pmm-store ./pmem.conf

	or

	./startomnisci --pmm ./vpmem.conf --pmm-store ./pmem.conf

  Option --pmm enables AppDirect volatile mode and -pmm-store enables AppDirect persistence mode.

## AppDirect Volatile Mode

  In this mode, Optane DCPMM is used as volatile memory for capacity (therefore, data stored in DCPMM does not survive OmniSciDB restarts). Specially, cold columns are placed in DCPMM and hot columns are placed in DRAM for best performance. Please refer to the white papre titled  "Enabling Intel(R) �Optane DC Persistent Memory for OmniSciDB" for how to identify hot and cold columns).

  To enable this mode, option --pmm is required when OmniSciDB is started. This option takes one argument which is a configuration file listing the DCPMM locations in the file system and sizes for volatile usage. Each line of the file is one DCPMM pool, for example:

	/mnt/pmem2	512
	/mnt/pmem4	0

  The first line means /mnt/pmem2 is a peristent memory pool and the maximum size for OmniSciDB in this pool will be 512GB. If the maximum available memory is less than 512GB, for example, 256GB, then 256GB will be allocated to OmniSciDB.
  The second line means /mnt/pmem4 is a persistent memory pool and all availabe memory will be allocated to OmniSciDB.

## AppDirect Persistence Mode

  In this mode, Optane DCPMM is used as persistent storage. Columns are stored in DCPMM instead of hard disks or SSDs for performance (String dictionaries are still stored in disks). To enable this mode, option --pmm-store is required when OmniSciDB is started. Like --pmm option, --pmm-store also takes one argument which is a configuration file listing the DCPMM locations in the file system and sizes for persistent storage usage. Each line of the file is one DCPMM pool, for example:

	/mnt/pmem1	512
	/mnt/pmem3	0

  The first line means /mnt/pmem1 is a peristent memory pool and the maximum size for OmniSciDB in this pool will be 512GB. If the maximum available memory is less than 512GB, for example, 256GB, then 256GB will be allocated to OmniSciDB.
  The second line means /mnt/pmem3 is a persistent memory pool and all availabe memory will be allocated to OmniSciDB.

  Different from volatile usage, the locations and sizes can not change once they are put into use. However, new memory pools can always be added later.

## Combining Both AppDirect Volaile Mode and AppDirect Persistence Mode

  If you have a large capacity of DCPMM, you may want to partition the DCPMM to use part for volatile memory and the other part for storage with careful planning and calculations. This is very feasible and you can easily do this with both --pmm and --pmm-store options enabled at the same time. One benefit of this usage is that cold columns will be directly accessed from the DCPMM pools for storage though hot columns are still placed into DRAM from DCPMM storage.
