/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

#include "Shared/OpenSSLTypes.h"

class PkiEncryptor {
 public:
  /**
   * Generates a new encryption certificate at the set key store path,
   * if one does not already exist. Generated certificate file is only
   * readable and writable by the owner of the server process.
   */
  static void generateEncryptionCertificateIfNotExists();

  /**
   * Extracts the public key from encryption certificate at the set key store path
   * and encrypts provided plain text using this key.
   *
   * @param plain_text - plain text that will be encrypted using public key from
   * encryption certificate, length is not bound.
   * @return A serialized series of cipher text segments resulting from encryption
   */
  static std::string publicKeyEncrypt(const std::string& plain_text);

  /**
   * Extracts the private key from encryption certificate at the set key store path
   * and decrypts provided cipher text using this key.
   *
   * @param cipher_text - A serialized series of cipher texts segments that will be
   * decrypted using private key from encryption certificate
   * @return plain text resulting from decryption
   */
  static std::string privateKeyDecrypt(
      const std::string& serialized_cipher_text_segments);

  /**
   * Sets the key store path that will contain the certificate used for encryption
   * and decryption.
   *
   * @param key_store_path - path to key store
   */
  static void setKeyStorePath(const std::string& key_store_path) {
    key_store_path_ = key_store_path;
  }

  // MAX_PLAIN_TEXT_SEGMENT_WIDTH exceeding 214 results in rsa encryption exceptions
  inline static const size_t MAX_PLAIN_TEXT_SEGMENT_WIDTH = 200;
  inline static const size_t CIPHER_TEXT_SEGMENT_WIDTH = 256;

 private:
  static std::string getCertificatePath();

  static std::string key_store_path_;

  static std::string publicKeyEncryptSegment(
      const std::unique_ptr<EVP_PKEY_CTX, std::function<void(EVP_PKEY_CTX*)>>& ctx,
      const std::string& plain_text);

  static std::string privateKeyDecryptSegment(
      const std::unique_ptr<EVP_PKEY_CTX, std::function<void(EVP_PKEY_CTX*)>>& ctx,
      const std::string& cipher_text);

  static std::vector<std::string> deserializeEncryptedSegments(
      const std::string& serialized_cipher_text_segments);
};
