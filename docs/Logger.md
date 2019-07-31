# OmniSci Logger

The OmniSci logger is written in C++ and located at [Shared/Logger.{h,cpp}](../Shared/Logger.h).  It is based on Boost.Logger with a design goal of being backward compatible with Glog. The motivation for the change from Glog is customization over the format of the logging which was not possible with Glog's API (without source modification). Ancillary benefits include greater control over standard logging features.

## Quick Start

```
#include "Shared/Logger.h"
```
 * Initialize a `LogOptions` object. E.g.<br/>
   `logger::LogOptions log_options(argv[0]);`
 * `LogOptions` can optionally be added to `boost::program_options`:<br/>
   `help_desc.add(log_options.get_options());`
 * Initialize global logger once per application:
   `logger::init(log_options);`
 * From anywhere in the program:
    * `LOG(INFO) << "Nice query!";`
    * `LOG(DEBUG4) << "x = " << x;`
    * `CHECK(condition);`
    * `CHECK_LE(x, xmax);`

Newlines are automatically appended to log messages.

## Additional Notes

Boost.Log is a flexible logging system with sources, a central core, and sinks. See https://www.boost.org/libs/log/doc/html/log/design.html for an overview.

`omnisci_server --help` includes the logging options:
```
Logging:
--log-directory arg (="mapd_log")  Logging directory. May be relative to
                                      data directory, or absolute.
--log-file-name arg (=omnisci_server.{SEVERITY}.%Y%m%d-%H%M%S.log)
                                      Log file name relative to
                                      log-directory.
--log-symlink arg (=omnisci_server.{SEVERITY})
                                      Symlink to active log.
--log-severity arg (=INFO)            Log to file severity level: INFO
                                      WARNING ERROR FATAL
--log-severity-clog arg (=ERROR)      Log to console severity level: INFO
                                      WARNING ERROR FATAL
--log-channels arg                    Log channel debug info: IR PTX
--log-auto-flush arg (=1)             Flush logging buffer to file after each
                                      message.
--log-max-files arg (=100)            Maximum number of log files to keep.
--log-min-free-space arg (=20971520)  Minimum number of bytes left on device
                                      before oldest log files are deleted.
--log-rotate-daily arg (=1)           Start new log files at midnight.
--log-rotation-size arg (=10485760)   Maximum file size in bytes before new
                                      log files are started.
```

There are 8 logging levels, or severities:
 * `DEBUG4`
 * `DEBUG3`
 * `DEBUG2`
 * `DEBUG1`
 * `INFO`
 * `WARNING`
 * `ERROR`
 * `FATAL`

Specifying a log severity level in the config (e.g. `--log-severity=DEBUG2`) causes all log messages of the severity
of higher (below it in the above list) to be logged.  A separate log file is created for `INFO`, `WARNING`, `ERROR`,
and `FATAL`. Each file logs messages at its own level and higher. E.g. `WARNING` log file includes both `WARNING`
and `ERROR` messages, but `ERROR` log file does not include `WARNING` messages. The one exception to this is the
`INFO` log file which includes all active log levels. There are no separate files for the `DEBUG` levels. Instead,
all active `DEBUG` levels, if any, are logged into the `INFO` log file.

`abort()` is called whenever a severity of `FATAL` is logged.

`VLOG(n)` is replaced by `LOG(DEBUGn)`. Thus to see all `VLOG(1)` and `VLOG(2)` messages on the console, use `--log-severity-clog DEBUG2`.

There are 7 CHECK macros:

* `CHECK()`
* `CHECK_EQ()`
* `CHECK_NE()`
* `CHECK_LT()`
* `CHECK_LE()`
* `CHECK_GT()`
* `CHECK_GE()`

When a `CHECK*()` macro fails, a message is logged at the `FATAL` severity level and `abort()` is called.

`LOG_IF` is a conditional `LOG` that only logs a message when the condition is true.
Example: `LOG_IF(INFO, is_logged_in) << "User " << user << " is logged in."`

## Channels

In addition to the above severity levels, developers have the option of creating independent channels
outside of the Severity hierarchy. To add a new channel, add to the following 3 items in `Shared/Logger.h`:
1. `enum Channel` - A distinct enum, e.g. `FOO`.
2. `std::array ChannelNames` - The enum in string form, e.g. `"FOO"`.
3. `std::array ChannelSymbols` - A unique single-character that shows up in the logs, e.g. `'O'` (`'F'` is reserved by `FATAL`.)

One can then log to this channel just like a severity: `LOG(FOO) << message;`.

To activate the channel, it must be given via the program options. E.g. to activate channels `FOO` and `BAR`:

    omnisci_server --log-channels=FOO,BAR data

This will cause all `FOO` and `BAR` log entries to be logged into their own separately named files.

## Deprecated Options

One old glog-related option is deprecated:

* `flush-log` - replaced by `log-auto-flush`.

## Log File Format

The general format of log files is

```
(timestamp) (level) (process_id) (filename:line_number) (message)
```

MapDHandler in particular uses an extension called LogSession which provides additional structured information:

```
(timestamp) (level) (process_id) (filename:line_number) stdlog (function_name) (match_id) (time_ms) (username) (dbname) (public_session_id) (array of names) (array of values)
```

where

* `level` - 1 of 8 characters designating the log level: `4`, `3`, `2`, `1`, `I`, `W`, `E`, `F`. In channel logs, it is the 1-character symbol for the channel.
* `match_id` - integer value used to match up the `stdlog_begin`/`stdlog` pairs.
* `time_ms` - time in milliseconds between begin and end of function call.

If `DEBUG1` logging is enabled, then a corresponding `stdlog_begin` line is also logged.
