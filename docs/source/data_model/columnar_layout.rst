.. HeavyDB Data Model

==================================
Columnar Data Organization
==================================

Columns
==================
HeavyDB is a columnar database. Columns in HeavyDB are organized into fragments (striping across rows) and chunks (the intersection of a column and a fragment). The following sections describe fragment and chunk organization and layout in depth. 

Fragments
==================

.. include:: ./columnar_data_organization/fragment.inc

.. _chunk-label:

Chunks
==================

.. include:: ./columnar_data_organization/chunk.inc
