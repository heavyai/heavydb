set(shapelib_src
  shapelib/shpopen.c
  shapelib/shptree.c
  shapelib/dbfopen.c
  shapelib/safileio.c
)
set(shapelib_hdr
  shapelib/shapefil.h
)

add_library(shapelib ${shapelib_src} ${shapelib_hdr})
