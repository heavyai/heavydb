.. OmniSciDB documentation master file, created by
   sphinx-quickstart on Sun Feb 10 09:03:31 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

OmniSciDB Developer Documentation
========================================

.. toctree::
    :caption: High Level Overview
    :maxdepth: 2
    :numbered:

    overview/index

.. toctree::
    :caption: Data Model
    :maxdepth: 3
    :numbered:

    data_model/columnar_layout
    data_model/physical_layout
    data_model/Catalog/index.rst
    data_model/memory_layout
    data_model/api
    data_model/types

.. toctree::
    :caption: Query Execution
    :maxdepth: 2
    :numbered:

    execution/flow
    execution/parse
    execution/optimization
    execution/scheduler
    execution/codegen
    execution/kernels
    execution/results

.. toctree::
    :caption: Components
    :maxdepth: 2
    :numbered:

    components/logger
    components/query_state
    components/thrift_api



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
