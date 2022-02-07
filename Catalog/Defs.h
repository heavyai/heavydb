#pragma once

#include <limits>

// SPI means Sequential Positional Index which is equivalent to the input index in a
// RexInput node
#define SPIMAP_MAGIC1 (std::numeric_limits<unsigned>::max() / 4)
#define SPIMAP_MAGIC2 8
#define SPIMAP_GEO_PHYSICAL_INPUT(c, i) \
  (SPIMAP_MAGIC1 + (unsigned)(SPIMAP_MAGIC2 * ((c) + 1) + (i)))
