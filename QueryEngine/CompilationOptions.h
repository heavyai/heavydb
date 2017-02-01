#ifndef QUERYENGINE_COMPILATIONOPTIONS_H
#define QUERYENGINE_COMPILATIONOPTIONS_H

enum class ExecutorDeviceType { CPU, GPU, Hybrid };

enum class ExecutorOptLevel { Default, LoopStrengthReduction };

struct CompilationOptions {
  ExecutorDeviceType device_type_;
  const bool hoist_literals_;
  const ExecutorOptLevel opt_level_;
  const bool with_dynamic_watchdog_;
};

struct ExecutionOptions {
  const bool output_columnar_hint;
  const bool allow_multifrag;
  const bool just_explain;
  const bool allow_loop_joins;
  const bool with_watchdog;  // Per work unit, not global.
  const bool jit_debug;
  const bool just_validate;
  const bool with_dynamic_watchdog;            // Per work unit, not global.
  const unsigned dynamic_watchdog_time_limit;  // Dynamic watchdog time limit, in milliseconds.
};

#endif  // QUERYENGINE_COMPILATIONOPTIONS_H
