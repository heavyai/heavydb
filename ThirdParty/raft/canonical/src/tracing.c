#include "tracing.h"

/* No-op trace emit function. */
static inline void noopTracerEmit(struct raft_tracer *t,
                                  const char *file,
                                  int line,
                                  const char *message)
{
    (void)t;
    (void)file;
    (void)line;
    (void)message;
}

/* Default no-op tracer. */
struct raft_tracer NoopTracer = {.impl = NULL, .emit = noopTracerEmit};
