set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -Wbad-function-cast -Wcast-align -Wcast-qual -Wmissing-prototypes -Wstrict-prototypes -Wshadow -Wundef -Wpointer-arith -O2 -fomit-frame-pointer -funroll-loops")

set(bcrypt_src
  crypt_blowfish/crypt_blowfish.c
  crypt_blowfish/crypt_gensalt.c
  crypt_blowfish/wrapper.c
  crypt_blowfish/x86.S
  bcrypt.c
)

set(bcrypt_hdr
    bcrypt.h
)

add_library(bcrypt ${bcrypt_src} ${bcrypt_hdr})
