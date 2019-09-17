.. OmniSciDB Architecture Overview

==================================
OmnisciDB at 30,000 feet
==================================

High Level Diagram
==================

.. image:: ../img/platform_overview.png

Simple Execution Model
======================

.. uml::

    @startuml
   
    start
   
    :Parse and Validate SQL;
   
    :Generate Optimized 
     Relational Algebra Sequence;
   
    :Prepare Execution Environment;
    
    repeat
        fork
            :Data Ownership, 
             Identification, 
             Load (as required);
            :Execute Query Kernel 
             on Target Device;
        fork again
            :Data Ownership, 
             Identification, 
             Load (as required);
            :Execute Query Kernel 
             on Target Device;
        fork again
            :Data Ownership, 
             Identification, 
             Load (as required);
            :Execute Query Kernel 
             on Target Device;
        end fork      
        :Reduce Result;

    while (Query Completed?)

    :Return Result;
    
    stop

    @enduml

.. image:: ../img/query_exe_workflow.png
