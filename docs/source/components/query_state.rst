==========
QueryState
==========

``#include "ThriftHandler/QueryState.h"``

QueryState is a set of C++ classes that:

 - Consolidates information about current and past SQL queries.
 - Record execution time of blocks of C++ code, and log with call stack relationships.

Classes
-------

.. image:: ../img/query_state/classes.svg
   :align: center
   :alt: QueryState Classes

:QueryStates: Manages and aggregates ``QueryState`` objects. E.g. ``MapDHandler`` has one ``QueryStates`` member
   variable.
:QueryState: Manages the lifetime of a single query. Attributes/methods (e.g. ``bool just_explain``) may be
   moved/added directly to this class as needed to model the state of SQL queries. As such, the development
   of this class can be considered ongoing, as a reflection of the ongoing development of the OmniSciDB itself.
   Timing of code blocks (i.e. function calls) are stored in a list of ``Event``\ s.
:Event: Records start and stop times for the lifetime of a corresponding ``Timer`` object. ``Event``\ s can have
   parent/child relationships to model call-stack nesting.
:Timer: ``Event``\ s and ``Timer``\ s are created at the same time. A ``Timer`` object is typically instantiated
   with a ``__func__`` parameter, which shows up in the logs along with the duration between its construction
   and destruction.
:QueryStateProxy: A light-weight object that can both create and be created by ``Timer`` objects so
   that call-stack relationships between parent/child ``Event``\ s are recorded.

Example Usage
-------------

In ``MapDHandler::sql_execute()``::

    // session_ptr and query_str are already set or passed into the function.

    auto query_state = create_query_state(session_ptr, query_str); // Line A
    auto stdlog = STDLOG(query_state);                             // Line B
    ...
    foo(query_state->createQueryStateProxy());                     // Line C

In ``foo(QueryStateProxy query_state_proxy)``::

    auto timer = query_state_proxy.createTimer(__func__);          // Line D
    ...
    bar(timer.createQueryStateProxy());                            // Line E

In ``bar(QueryStateProxy query_state_proxy)``::

    auto timer = query_state_proxy.createTimer(__func__);          // Line F
    ...

:Line A: ``auto query_state = create_query_state(session_ptr, query_str); // Line A``

  Create a new ``QueryState`` in the ``MapDHandler::query_states_`` member variable for the current query.

:Line B: ``auto stdlog = STDLOG(query_state);                             // Line B``

  Connects the ``QueryState`` object to the ``stdlog`` which will log the ``QueryState`` information during
  ``stdlog``'s destructor.

:Line C: ``foo(query_state->createQueryStateProxy());                     // Line C``

  To pass ``QueryState`` to other functions, it is preferred to generate a light-weight ``QueryStateProxy``
  object rather than pass the ``QueryState`` reference. This is because ``Timer`` objects also generate
  ``QueryStateProxy`` objects to pass to other functions in order to track nested timings. Thus to avoid having
  redundant versions of functions that accept both ``QueryState`` and ``QueryStateProxy`` types, it is preferred
  for members/functions to accept ``QueryStateProxy``. The original ``QueryState`` reference is then available
  via ``QueryStateProxy::getQueryState()``.

:Line D: ``auto timer = query_state_proxy.createTimer(__func__);          // Line D``

  The lifetime of ``timer`` is recorded and stored in the original ``query_state`` object from
  Line A, and will appear in the logs associated with ``__func__ = "foo"`` in this case.

:Line E: ``bar(timer.createQueryStateProxy());                            // Line E``

  Similar to Line C, this is how to create nested timings when invoking another function deeper
  in the call stack.

:Line F: ``auto timer = query_state_proxy.createTimer(__func__);          // Line F``

  Similar to Line D, the created ``timer`` will show up nested under the previous ``Timer`` object.

The two ``timer`` instances above directly result in the nested log lines::

      foo 140038217238272 - total time 578 ms
        bar 140038217238272 - total time 578 ms

(The long integer is the thread id.) Though only a couple examples were given here, each ``Timer`` object can
spawn any number of ``QueryStateProxy`` instances, and vice-versa.

Summary
-------

The above example demonstrates how to:

1. Create new ``QueryState`` objects and connect them with ``SessionInfo`` and ``StdLog`` instances.
2. Create nested ``Timer`` objects using ``QueryStateProxy`` objects as intermediaries to model nested
   function calls.
