set(poly2tri_src
  poly2tri/common/shapes.cc
  poly2tri/sweep/advancing_front.cc
  poly2tri/sweep/cdt.cc
  poly2tri/sweep/sweep.cc
  poly2tri/sweep/sweep_context.cc
)
set(poly2tri_hdr
  poly2tri/poly2tri.h
  poly2tri/common/shapes.h
  poly2tri/common/utils.h
  poly2tri/sweep/advancing_front.h
  poly2tri/sweep/cdt.h
  poly2tri/sweep/sweep.h
  poly2tri/sweep/sweep_context.h
)

add_library(poly2tri ${poly2tri_src} ${poly2tri_hdr})
