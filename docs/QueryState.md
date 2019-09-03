# QueryState

QueryState is a set of C++ classes that:
 - Consolidates information about current and past queries.
 - Time and record blocks of C++ code, and log with call stack relationships.

## Classes

![QueryState Classes](images/query_state/classes.svg)

 - **QueryStates** - Manages and aggregates QueryState objects. E.g. `MapDHandler` has one `QueryStates` member
   variable.
 - **QueryState** - Manages the lifetime of a single query. Timing of code blocks (i.e. function calls) are stored
   in a list of Events.
 - **Event** - Records start and stop times for the lifetime of a corresponding Timer object. Events can have
   parent/child relationships to model call-stack nesting. More on this below.
 - **Timer** - Events and Timers are created together, and the `Event::start_` value is set to its creation time.
   The `Timer` holds an iterator to the corresponding `Event`, and calls `Event::stop()` from `~Timer()` which sets
   the `Event::stop_` value to the time of destruction. In other words, by creating a `Timer` object at the beginning
   of a block of code, the corresponding `Event` will record the amount of time the block took to execute.
 - **QueryStateProxy** - A light-weight object that can both create and be created by `Timer` objects so
   that call-stack relationships between parent/child Events are recorded.

## Example Usage

When `omnisci_server` is run with either `--verbose` or `--log-severity=DEBUG1` then the following stacked timings
show up in the log:

    2019-07-09T14:49:24.574086 1 17983 MapDHandler.cpp:776 stacked_times sql_execute 2 total time 828 ms
      parse_to_ra 140038217238272 - total time 578 ms
        processImpl 140038217238272 - total time 578 ms
      execute_rel_alg 140038217238272 - total time 250 ms
        convert_rows 140038217238272 - total time 0 ms

How to generate the nested `parse_to_ra()` and `processImpl` timings:

In `MapDHandler::sql_execute()`:

    auto session_ptr = get_session_ptr(session);                   // Line 1
    auto query_state = create_query_state(session_ptr, query_str); // Line 2
    auto stdlog = STDLOG(query_state);                             // Line 3
    ...
    sql_execute_impl(_return,
                     query_state->createQueryStateProxy(),
                     column_format,
                     nonce,
                     session_info.get_executor_device_type(),
                     first_n,
                     at_most_n);

1. Line 1: When associating `SessionInfo` with a `QueryState`, always use a `std::shared_ptr` copied from the
   "original" `SessionInfo`. In the case of `MapDHandler`, it is copied from `SessionMap sessions_` member variable.
   This is due to the fact that a `QueryState`'s lifetime can extend beyond a `SessionInfo`, thus `QueryState`
   will hold a corresponding `std::weak_ptr` based on the `std::shared_ptr` it is given.
2. Line 2: Create a new `QueryState` in the `QueryStates query_states_` member variable for the current query.
3. Line 3: Connects the `QueryState` object to the `stdlog` which will log the `QueryState` information during
   `stdlog`'s destructor.

The call to `sql_execute_impl()` shows how `QueryState` should generally be propagated to other functions, which
is via `QueryStateProxy` by value.

In `sql_execute_impl()`:

    query_ra = parse_to_ra(query_state_proxy, query_str, {}, tableNames, mapd_parameters_);

In `parse_to_ra()`:

    auto timer = query_state_proxy.createTimer(__func__);
    ...
    auto result = calcite_->process(timer.createQueryStateProxy(),
                                    legacy_syntax_ ? pg_shim(actual_query) : actual_query,
                                    filter_push_down_info,
                                    legacy_syntax_,
                                    pw.isCalciteExplain(),
                                    mapd_parameters.enable_calcite_view_optimize);

In `Calcite::process()` -> `Calcite::processImpl()`:

    auto timer = query_state_proxy.createTimer(__func__);

The two `timer` instances above directly result in the nested log lines:

      parse_to_ra 140038217238272 - total time 578 ms
        processImpl 140038217238272 - total time 578 ms

### Summary

What has been demonstrated in the above example is:

1. How to create new `QueryState` objects and connect them with `SessionInfo` and `StdLog` instances.
2. How to create nested `Timer` objects using `QueryStateProxy` objects as intermediaries to model nested
   function calls.
