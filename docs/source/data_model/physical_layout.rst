.. OmniSciDB Data Model

==================================
Physical Data Layout
==================================
The OmniSci database uses its file system layer to manage the persistence of data across system restart or server failure.

The database uses the AbstractBufferMgr paradigm, to treat the data on disks as chunks in a similar manner to the other data layers. However, unlike the other data storage layers in the database, the data on disk is stored not stored in single contiguous unit, rather at the physical disk layer it is stored across multiple ‘pages’ across multiple files.

The database manages disk storage via the `FileMgr` class. The system creates a single global file manager as the entry point for all file management, this object in turn has many file managers -  one per current table (see diagram :ref:`file-manager-structure`)

.. figure:: /img/DataMgr.png
   :name: file-manager-structure
   :align: center

   File Manager Object Hierarchy

Directory Structure
===================

.. include:: ./physical_data_layout/directory_structure.inc

Data Multipages
==================

.. include:: ./physical_data_layout/data_multipages.inc

Metadata Pages
==================

.. include:: ./physical_data_layout/metadata_pages.inc

