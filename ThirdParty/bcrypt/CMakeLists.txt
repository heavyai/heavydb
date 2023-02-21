set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -Wbad-function-cast -Wcast-align -Wcast-qual -Wmissing-prototypes -Wstrict-prototypes -Wshadow -Wundef -Wpointer-arith -O2 -fomit-frame-pointer -funroll-loops")

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /permissive /D WIN32_LEAN_AND_MEAN")
endif()

set(bcrypt_src
  crypt_blowfish/crypt_blowfish.c
  crypt_blowfish/crypt_gensalt.c
  crypt_blowfish/wrapper.c
  crypt_blowfish/x86.S
  bcrypt.c
)

set(bcrypt_hdr
    include/bcrypt.h
)

#The library name has been changed to rg3_bcrypt to prevent a library name conflict on ms windows
add_library(rg3_bcrypt ${bcrypt_src} ${bcrypt_hdr})
