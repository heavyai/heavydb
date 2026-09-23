/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <functional>
#include <memory>

#include <openssl/evp.h>
#include <openssl/rsa.h>
#include <openssl/x509.h>
#include <openssl/x509_vfy.h>

// Enable the use of older/deprecated functions for older
// versions of OpenSSL that do not define these functions
#if (OPENSSL_VERSION_NUMBER <= 0x10100000L) /* OpenSSL 1.1.0+ */
#define ASN1_STRING_get0_data(x) ASN1_STRING_data(x)
#define X509_GET_NOT_BEFORE(x) X509_get_notBefore(x)
#define X509_GET_NOT_AFTER(x) X509_get_notAfter(x)
#else
#define X509_GET_NOT_BEFORE(x) X509_getm_notBefore(x)
#define X509_GET_NOT_AFTER(x) X509_getm_notAfter(x)
#endif

using EVP_PKEY_ptr = std::unique_ptr<EVP_PKEY, std::function<void(EVP_PKEY*)>>;
inline void free_evp(EVP_PKEY* pkey) {
  if (nullptr != pkey) {
    EVP_PKEY_free(pkey);
  }
}

using RSA_ptr = std::unique_ptr<RSA, std::function<void(RSA*)>>;
inline void free_rsa(RSA* rsa) {
  if (nullptr != rsa) {
    RSA_free(rsa);
  }
}

using X509_ptr = std::unique_ptr<X509, std::function<void(X509*)>>;
inline void x509_deleter(X509* x509) {
  if (x509 != nullptr) {
    X509_free(x509);
  }
}

using X509_STORE_ptr = std::unique_ptr<X509_STORE, std::function<void(X509_STORE*)>>;
inline void x509_store_deleter(X509_STORE* x509_store) {
  if (x509_store != nullptr) {
    X509_STORE_free(x509_store);
  }
}
