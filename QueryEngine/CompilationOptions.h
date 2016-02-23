#ifndef QUERYENGINE_COMPILATIONOPTIONS_H
#define QUERYENGINE_COMPILATIONOPTIONS_H

enum class ExecutorDeviceType { CPU, GPU, Hybrid };

enum class ExecutorOptLevel { Default, LoopStrengthReduction };

struct CompilationOptions {
  ExecutorDeviceType device_type_;
  const bool hoist_literals_;
  const ExecutorOptLevel opt_level_;
};

#endif  // QUERYENGINE_COMPILATIONOPTIONS_H
