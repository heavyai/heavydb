/*
 *
 * This file is based on work at https://github.com/Thalhammer/jwt-cpp.
 *
 * MIT License
 *
 * Copyright (c) 2018 Dominik Thalhammer

 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *
 */
#include "cpp_jwt/ec_processor.h"
#include <openssl/bio.h>
#include <openssl/ec.h>
#include <openssl/ecdsa.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/pem.h>

#include <assert.h>
#include <algorithm>
#include <array>
#include <functional>
#include <memory>
#include <vector>

#include "Shared/base64.h"
// OpenSSL on windows doens't seem to have this in a header.
#if WIN32
typedef struct ECDSA_SIG_st {
  BIGNUM* r;
  BIGNUM* s;
} ECDSA_SIG;
#endif
/*
 * The following functions are a modified version of the EC processing
 * in algorithm.ipp in the project CPP-JWT.
 * (see https://github.com/arun11299/cpp-jwt)
 *
 * MIT License
 *
 * Copyright (c) 2017 Arun Muralidharan
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 */

std::string getError(const std::string& errLabel) {
  unsigned long errNo = ERR_get_error();
  std::array<char, 200> errBuf;
  ERR_error_string_n(errNo, &errBuf[0], errBuf.size());
  return errLabel + " [" + std::to_string(errNo) + "] " + errBuf[0];
}

using U_PKEY = std::unique_ptr<EVP_PKEY, std::function<void(EVP_PKEY*)>>;

void reprocessEcSignature(U_PKEY& pkey, std::string& decSig) {
  using U_EC_SIG = std::unique_ptr<ECDSA_SIG_st, std::function<void(ECDSA_SIG_st*)>>;
  U_EC_SIG ecSig(ECDSA_SIG_new(), [](ECDSA_SIG_st* ptr) {
    if (ptr)
      ECDSA_SIG_free(ptr);
  });
  if (ecSig == nullptr)
    throw(std::runtime_error(getError("Error in ECDSA_SIG_new")));

  using U_EC_KEY = std::unique_ptr<EC_KEY, std::function<void(EC_KEY*)>>;
  U_EC_KEY ec_key(EVP_PKEY_get1_EC_KEY(pkey.get()), [](EC_KEY* ptr) {
    if (ptr)
      EC_KEY_free(ptr);
  });
  if (!ec_key)
    throw(std::runtime_error(getError("Error in EVP_PKEY_get1_EC_KEY")));

  unsigned int degree = EC_GROUP_get_degree(EC_KEY_get0_group(ec_key.get()));
  unsigned int bn_len = (degree + 7) / 8;
  if ((bn_len * 2) != decSig.length()) {
    std::string err = "Error in EC degree calculation [" + std::to_string(bn_len) + ":" +
                      std::to_string(decSig.length()) + "]";
    throw(std::runtime_error(getError(err)));
  }

  // These BIGNUMs will be deleted via ecSig
  BIGNUM* ecSigR = BN_bin2bn((unsigned char*)decSig.data(), bn_len, nullptr);
  if (!ecSigR)
    throw(std::runtime_error(getError("Error in BN_bin2bn r")));
  BIGNUM* ecSigS = BN_bin2bn((unsigned char*)decSig.data() + bn_len, bn_len, nullptr);
  if (!ecSigS) {
    // need to manually free ecSigR as it hasn't been assigned to ecSig yet.
    BN_clear_free(ecSigR);
    throw(std::runtime_error(getError("Error in BN_bin2bn s")));
  }
  BN_clear_free(ecSig->r);
  BN_clear_free(ecSig->s);
  ecSig->r = ecSigR;
  ecSig->s = ecSigS;

  size_t nlen = i2d_ECDSA_SIG(ecSig.get(), nullptr);
  decSig.resize(nlen);
  auto data = reinterpret_cast<unsigned char*>(&decSig[0]);
  nlen = i2d_ECDSA_SIG(ecSig.get(), &data);
  if (nlen == 0)
    throw(std::runtime_error(getError("i2d_ECDSA_SIG")));
}

bool ec_verify(const std::string& payload,
               const std::string& signature,
               const std::string& ecPubKey) {
  // In some circumstances the signature needs to be changed in this
  // function
  std::string decSig = signature;

  // 1. Get a BIO - basic IO and load public key.
  using U_BIO = std::unique_ptr<BIO, std::function<void(BIO*)>>;
  U_BIO bioPub(BIO_new_mem_buf((void*)ecPubKey.c_str(), (int)ecPubKey.length()),
               [](BIO* ptr) {
                 if (ptr)
                   BIO_free_all(ptr);
               });
  if (!bioPub)
    throw(std::runtime_error(getError("Error in BIO_new_men_buf")));

  // 2. Extract pubkey from bio,
  U_PKEY pubKey(PEM_read_bio_PUBKEY(bioPub.get(), nullptr, nullptr, nullptr),
                [](EVP_PKEY* ptr) {
                  if (ptr)
                    EVP_PKEY_free(ptr);
                });
  if (!pubKey)
    throw(std::runtime_error(getError("Error in PEM_read_bio_PUBKEY")));

  // 3. If the public key is in EC format we need to re-work signature.
  if (EVP_PKEY_EC == EVP_PKEY_id(pubKey.get())) {
    reprocessEcSignature(pubKey, decSig);
  }

  using U_EVP_MD_CTX = std::unique_ptr<EVP_MD_CTX, std::function<void(EVP_MD_CTX*)>>;

  U_EVP_MD_CTX mdctx(EVP_MD_CTX_create(), [](EVP_MD_CTX* ptr) {
    if (ptr)
      EVP_MD_CTX_destroy(ptr);
  });
  if (!mdctx)
    throw(std::runtime_error(getError("Error in EVP_MD_CTX_create")));

  if (1 != EVP_DigestVerifyInit(mdctx.get(), NULL, EVP_sha512(), NULL, pubKey.get()))
    throw(std::runtime_error(getError("Error in EVP_DigestVerifyInit")));

  if (1 != EVP_DigestVerifyUpdate(mdctx.get(), payload.c_str(), payload.length()))
    throw(std::runtime_error(getError("Error in EVP_DigestVerifyVerifyUpdate")));

  int rc =
      EVP_DigestVerifyFinal(mdctx.get(), (unsigned char*)&decSig[0], decSig.length());
  // returns 1 for valid signature 0 for invalid signature, anyting else is an
  // error.
  if (rc == 1 || rc == 0)
    return (rc == 1);
  throw(std::runtime_error(getError("Error in EVP_DigestVerifyVerifyFinal")));
}
