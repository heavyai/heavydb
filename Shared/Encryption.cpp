/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Encryption.h"

#include <openssl/asn1.h>
#include <openssl/err.h>
#include <openssl/pem.h>

#include "Logger/Logger.h"
#include "Shared/OpenSSLTypes.h"
#include "Shared/SysDefinitions.h"

#include <boost/filesystem.hpp>

namespace {
constexpr int RSA_MODULUS_SIZE = 2048;
constexpr long CERT_EXPIRATION_DURATION_IN_SECONDS = 365 * 24 * 60 * 60;

void log_openssl_error(const std::string& error_log_prefix) {
  std::unique_ptr<BIO, std::function<void(BIO*)>> bio = {
      BIO_new(BIO_s_mem()), [](BIO* ptr) { BIO_free_all(ptr); }};
  ERR_print_errors(bio.get());
  char* openssl_error = nullptr;
  size_t length = BIO_get_mem_data(bio.get(), &openssl_error);
  LOG(ERROR) << error_log_prefix
             << ". OpenSSL error: " << std::string{openssl_error, length};
}

void throw_cert_generation_exception(const std::string& error_log_prefix) {
  log_openssl_error(error_log_prefix);
  throw std::runtime_error{
      "An error occurred when attempting to generate a new encryption certificate."};
}

void throw_encryption_exception(const std::string& error_log_prefix) {
  log_openssl_error(error_log_prefix);
  throw std::runtime_error{"An error occurred when attempting to encrypt plain text."};
}

void throw_decryption_exception(const std::string& error_log_prefix) {
  log_openssl_error(error_log_prefix);
  throw std::runtime_error{"An error occurred when attempting to decrypt cipher text."};
}

bool is_expired_cert(const X509* x509) {
  bool is_expired{false};
  int days, seconds;
  if (ASN1_TIME_diff(&days, &seconds, NULL, X509_GET_NOT_AFTER(x509)) == 1) {
    is_expired = (days < 0 || seconds < 0);
  }
  return is_expired;
}
}  // namespace

void PkiEncryptor::generateEncryptionCertificateIfNotExists() {
  const auto cert_file_path = getCertificatePath();
  if (boost::filesystem::exists(cert_file_path)) {
    return;
  }

  EVP_PKEY_ptr pkey = {EVP_PKEY_new(), free_evp};
  std::unique_ptr<BIGNUM, std::function<void(BIGNUM*)>> exponent = {
      BN_new(), [](BIGNUM* ptr) { BN_free(ptr); }};
  if (BN_set_word(exponent.get(), RSA_F4) != 1) {
    throw_cert_generation_exception("Failed to set RSA public exponent.");
  }

  RSA_ptr rsa_unique_ptr = {RSA_new(), free_rsa};
  RSA* rsa = rsa_unique_ptr.get();
  if (RSA_generate_key_ex(rsa, RSA_MODULUS_SIZE, exponent.get(), NULL) != 1) {
    throw_cert_generation_exception("Failed to generate RSA key pair.");
  }
  EVP_PKEY_assign_RSA(pkey.get(), rsa);
  // `EVP_PKEY_assign_RSA` ensures that `rsa` is freed when `pkey` is freed.
  // Releasing ownership of `rsa` below in order to avoid a double delete.
  rsa_unique_ptr.release();
  X509_ptr x509 = {X509_new(), x509_deleter};

  X509_set_pubkey(x509.get(), pkey.get());
  X509_gmtime_adj(X509_GET_NOT_BEFORE(x509.get()), 0);
  X509_gmtime_adj(X509_GET_NOT_AFTER(x509.get()), CERT_EXPIRATION_DURATION_IN_SECONDS);
  X509_sign(x509.get(), pkey.get(), EVP_sha256());

  std::unique_ptr<BIO, std::function<void(BIO*)>> cert_file = {
      BIO_new_file(cert_file_path.c_str(), "w+"), [](BIO* ptr) { BIO_free_all(ptr); }};
  if (PEM_write_bio_X509(cert_file.get(), x509.get()) != 1) {
    throw_cert_generation_exception("Failed to write x509 certificate to file.");
  }

  if (PEM_write_bio_RSAPrivateKey(cert_file.get(), rsa, NULL, NULL, 0, NULL, NULL) != 1) {
    throw_cert_generation_exception("Failed to write RSA private key to file.");
  }
  boost::filesystem::permissions(cert_file_path,
                                 boost::filesystem::perms::group_all |
                                     boost::filesystem::perms::others_all |
                                     boost::filesystem::perms::remove_perms);
}

std::string PkiEncryptor::publicKeyEncrypt(const std::string& plain_text) {
  const auto cert_file_path = getCertificatePath();
  CHECK(boost::filesystem::exists(cert_file_path));
  std::unique_ptr<BIO, std::function<void(BIO*)>> cert_file = {
      BIO_new(BIO_s_file()), [](BIO* ptr) { BIO_free_all(ptr); }};
  if (BIO_read_filename(cert_file.get(), cert_file_path.c_str()) != 1) {
    throw_encryption_exception("Failed to read file name " + cert_file_path);
  }

  X509_ptr x509 = {PEM_read_bio_X509(cert_file.get(), NULL, 0, NULL), x509_deleter};
  if (!x509) {
    throw_encryption_exception("Failed to read x509 certificate from path " +
                               cert_file_path);
  }

  if (is_expired_cert(x509.get())) {
    LOG(WARNING) << "Encrypting plain text using key from expired certificate.";
  }

  EVP_PKEY_ptr pkey = {X509_get_pubkey(x509.get()), free_evp};
  if (!pkey) {
    throw_encryption_exception("Failed to read public key from x509 certificate.");
  }

  std::unique_ptr<EVP_PKEY_CTX, std::function<void(EVP_PKEY_CTX*)>> ctx = {
      EVP_PKEY_CTX_new(pkey.get(), NULL),
      [](EVP_PKEY_CTX* ptr) { EVP_PKEY_CTX_free(ptr); }};
  if (!ctx) {
    throw_encryption_exception(
        "Failed to allocate public key algorithm context for encryption.");
  }

  if (EVP_PKEY_encrypt_init(ctx.get()) <= 0) {
    throw_encryption_exception(
        "Failed to initialize public key algorithm context for encryption.");
  }

  if (EVP_PKEY_CTX_set_rsa_padding(ctx.get(), RSA_PKCS1_OAEP_PADDING) <= 0) {
    throw_encryption_exception("Failed to set RSA padding mode for encryption.");
  }

  std::string serialized_cipher_text_segments;
  for (size_t segment_start = 0; segment_start < plain_text.length();
       segment_start += MAX_PLAIN_TEXT_SEGMENT_WIDTH) {
    serialized_cipher_text_segments += publicKeyEncryptSegment(
        ctx, plain_text.substr(segment_start, MAX_PLAIN_TEXT_SEGMENT_WIDTH));
  }
  return serialized_cipher_text_segments;
}

std::string PkiEncryptor::publicKeyEncryptSegment(
    const std::unique_ptr<EVP_PKEY_CTX, std::function<void(EVP_PKEY_CTX*)>>& ctx,
    const std::string& plain_text) {
  CHECK(plain_text.length() <= PkiEncryptor::MAX_PLAIN_TEXT_SEGMENT_WIDTH)
      << "Cannot encrypt plain text segments longer than "
      << PkiEncryptor::MAX_PLAIN_TEXT_SEGMENT_WIDTH << " bytes.";

  size_t cipher_text_length;
  const unsigned char* plain_text_ptr =
      reinterpret_cast<const unsigned char*>(plain_text.c_str());
  if (EVP_PKEY_encrypt(
          ctx.get(), NULL, &cipher_text_length, plain_text_ptr, plain_text.length()) <=
      0) {
    throw_encryption_exception("Failed to get cipher text length.");
  }

  std::unique_ptr<char, std::function<void(char*)>> cipher_text = {
      static_cast<char*>(OPENSSL_malloc(cipher_text_length)),
      [](char* ptr) { OPENSSL_free(ptr); }};

  if (!cipher_text) {
    throw_encryption_exception("Failed to allocate cipher text.");
  }

  if (EVP_PKEY_encrypt(ctx.get(),
                       reinterpret_cast<unsigned char*>(cipher_text.get()),
                       &cipher_text_length,
                       plain_text_ptr,
                       plain_text.length()) <= 0) {
    throw_encryption_exception("Failed to encrypt plain text.");
  }

  CHECK(cipher_text_length == PkiEncryptor::CIPHER_TEXT_SEGMENT_WIDTH)
      << "Encrypting a plain text segment returned a cipher text segment of unexpected "
         "length.";
  return std::string{cipher_text.get(), cipher_text_length};
}

std::string PkiEncryptor::privateKeyDecrypt(
    const std::string& serialized_cipher_text_segments) {
  const auto cert_file_path = getCertificatePath();
  CHECK(boost::filesystem::exists(cert_file_path));
  std::unique_ptr<BIO, std::function<void(BIO*)>> cert_file = {BIO_new(BIO_s_file()),
                                                               [](BIO* ptr) {
                                                                 BIO_free_all(ptr);
                                                                 ;
                                                               }};
  if (BIO_read_filename(cert_file.get(), cert_file_path.c_str()) != 1) {
    throw_decryption_exception("Failed to read file name " + cert_file_path);
  }

  RSA_ptr rsa = {PEM_read_bio_RSAPrivateKey(cert_file.get(), NULL, NULL, NULL), free_rsa};
  if (!rsa) {
    throw_decryption_exception("Failed to read private key from file");
  }

  EVP_PKEY_ptr pkey = {EVP_PKEY_new(), free_evp};
  EVP_PKEY_assign_RSA(pkey.get(), rsa.get());
  rsa.release();

  std::unique_ptr<EVP_PKEY_CTX, std::function<void(EVP_PKEY_CTX*)>> ctx = {
      EVP_PKEY_CTX_new(pkey.get(), NULL),
      [](EVP_PKEY_CTX* ptr) { EVP_PKEY_CTX_free(ptr); }};
  if (!ctx) {
    throw_decryption_exception(
        "Failed to allocate public key algorithm context for decryption.");
  }

  if (EVP_PKEY_decrypt_init(ctx.get()) <= 0) {
    throw_decryption_exception(
        "Failed to initialize public key algorithm context for decryption.");
  }

  if (EVP_PKEY_CTX_set_rsa_padding(ctx.get(), RSA_PKCS1_OAEP_PADDING) <= 0) {
    throw_decryption_exception("Failed to set RSA padding mode for decryption.");
  }

  std::string plain_text;
  for (const auto& cipher_text :
       deserializeEncryptedSegments(serialized_cipher_text_segments)) {
    plain_text += privateKeyDecryptSegment(ctx, cipher_text);
  }
  return plain_text;
}

std::string PkiEncryptor::privateKeyDecryptSegment(
    const std::unique_ptr<EVP_PKEY_CTX, std::function<void(EVP_PKEY_CTX*)>>& ctx,
    const std::string& cipher_text) {
  CHECK(cipher_text.length() == PkiEncryptor::CIPHER_TEXT_SEGMENT_WIDTH)
      << "Cannot decrypt cipher text segments with length unequal to "
      << PkiEncryptor::CIPHER_TEXT_SEGMENT_WIDTH << " bytes.";

  size_t plain_text_length;
  if (EVP_PKEY_decrypt(ctx.get(),
                       NULL,
                       &plain_text_length,
                       reinterpret_cast<const unsigned char*>(cipher_text.c_str()),
                       cipher_text.length()) <= 0) {
    throw_decryption_exception("Failed to get plain text length.");
  }

  std::unique_ptr<char, std::function<void(char*)>> plain_text = {
      static_cast<char*>(OPENSSL_malloc(plain_text_length)),
      [](char* ptr) { OPENSSL_free(ptr); }};

  if (!plain_text) {
    throw_decryption_exception("Failed to allocate plain text.");
  }

  if (EVP_PKEY_decrypt(ctx.get(),
                       reinterpret_cast<unsigned char*>(plain_text.get()),
                       &plain_text_length,
                       reinterpret_cast<const unsigned char*>(cipher_text.c_str()),
                       cipher_text.length()) <= 0) {
    throw_decryption_exception("Failed to decrypt cipher text.");
  }

  return std::string{plain_text.get(), plain_text_length};
}

std::vector<std::string> PkiEncryptor::deserializeEncryptedSegments(
    const std::string& serialized_cipher_text_segments) {
  std::vector<std::string> segments;
  for (size_t segment_start = 0; segment_start < serialized_cipher_text_segments.length();
       segment_start += PkiEncryptor::CIPHER_TEXT_SEGMENT_WIDTH) {
    segments.push_back(serialized_cipher_text_segments.substr(
        segment_start, PkiEncryptor::CIPHER_TEXT_SEGMENT_WIDTH));
  }
  return segments;
}

std::string PkiEncryptor::getCertificatePath() {
  CHECK(!key_store_path_.empty());
  return key_store_path_ + "/" + shared::kDefaultKeyFileName;
}

std::string PkiEncryptor::key_store_path_ = {};
