/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file PkiEncryptorTest.cpp
 * @brief Test suite for PkiEncryptor
 *
 */

#include <gtest/gtest.h>
#include <boost/filesystem.hpp>

#include "Shared/Encryption.h"
#include "Shared/SysDefinitions.h"
#include "Tests/TestHelpers.h"

class PkiEncryptorTest : public testing::Test {
 protected:
  inline static const std::string GENERATED_KEY_STORE_PATH{"./TestCert"};
  inline static const std::string EXISTING_KEY_STORE_PATH{
      "../../Tests/Encryption/ValidCert/"};
  inline static const std::string EXPIRED_KEY_STORE_PATH{
      "../../Tests/Encryption/ExpiredCert/"};
  inline static const std::string MISMATCHED_KEYS_KEY_STORE_PATH{
      "../../Tests/Encryption/MismatchedKeys/"};
  inline static const std::string INVALID_CERT_KEY_STORE_PATH{
      "../../Tests/Encryption/InvalidCert/"};
  inline static const std::string TEST_PLAIN_TEXT{"test plain text"};
  inline static const std::string LEGACY_ENCRYPTED_CIPHER_TEXT{
      "U\244\335\066\301\357^\006\006\210\214\352r\336L0\372\207E\vh\276"
      "\027\363̤ʸ\375\306G0\312^\352>\a\204\265&9\341\267\363(\371\026\271Br"
      "\266\240\267L\335\v\256\233\004\373\017\035U\023Z\024\211`\276d\274-aK"
      "\376\vA|§<ݟ6*D\005}Hq\224\330\362\313\362\217\372q\262\251\213\006\335'z"
      "\233\v\331*\372\216js\353=v\276F\260\060\251g\277_\"]\316.Y\235U\332I\315"
      "\316+\253Q\032\021\061<j\302Ra7\257\026\266\243\024O\201\305\310\317P\f"
      "\210\061\220\364Ŏ,\267\333\063\316\356c/9AMl\321:\020ۡ\002&H\223\"W\336)"
      "UdF\242\273\217\222\266\365\305#\267㙟g\302\310\327\255D\326\036U\341"
      "\310Ɨ?G\354H\367gYKȹ!\263\026I\362q\361*-\370\325\307\314bz>\274\316^"
      "\204\274cp\276\313H\277L"};

  void TearDown() override {
    if (boost::filesystem::exists(GENERATED_KEY_STORE_PATH)) {
      boost::filesystem::remove_all(GENERATED_KEY_STORE_PATH);
    }
  }

  void assertNoPlainTextExists(const std::string& cipher_text) {
    ASSERT_TRUE(cipher_text.find("test") == std::string::npos);
    ASSERT_TRUE(cipher_text.find("plain") == std::string::npos);
    ASSERT_TRUE(cipher_text.find("text") == std::string::npos);
  }

  void assertException(std::function<void()> function, const std::string& error_message) {
    try {
      function();
      FAIL() << "An exception should have been thrown for this test case.";
    } catch (const std::exception& e) {
      ASSERT_EQ(error_message, e.what());
    }
  }

  std::string getRepeatedString(const std::string& val, const int repetitions) {
    std::string repeated_string;
    for (int i = 0; i < repetitions; i++) {
      repeated_string += val;
    }
    return repeated_string;
  }
};

TEST_F(PkiEncryptorTest, GenerateEncryptionCertificate) {
  PkiEncryptor::setKeyStorePath(GENERATED_KEY_STORE_PATH);
  boost::filesystem::create_directory(GENERATED_KEY_STORE_PATH);
  PkiEncryptor::generateEncryptionCertificateIfNotExists();
  const auto cert_path = GENERATED_KEY_STORE_PATH + "/" + shared::kDefaultKeyFileName;
  ASSERT_TRUE(boost::filesystem::exists(cert_path));

  const auto permissions = boost::filesystem::status(cert_path).permissions();
  ASSERT_EQ(permissions & boost::filesystem::perms::others_all,
            boost::filesystem::perms::no_perms);
  ASSERT_EQ(permissions & boost::filesystem::perms::group_all,
            boost::filesystem::perms::no_perms);
  ASSERT_EQ(permissions & boost::filesystem::perms::owner_exe,
            boost::filesystem::perms::no_perms);
  ASSERT_EQ(permissions & boost::filesystem::perms::owner_read,
            boost::filesystem::perms::owner_read);
  ASSERT_EQ(permissions & boost::filesystem::perms::owner_write,
            boost::filesystem::perms::owner_write);
}

TEST_F(PkiEncryptorTest, EncryptAndDecryptWithGeneratedCert) {
  PkiEncryptor::setKeyStorePath(GENERATED_KEY_STORE_PATH);
  boost::filesystem::create_directory(GENERATED_KEY_STORE_PATH);
  PkiEncryptor::generateEncryptionCertificateIfNotExists();

  const std::string cipher_text = PkiEncryptor::publicKeyEncrypt(TEST_PLAIN_TEXT);
  assertNoPlainTextExists(cipher_text);

  const std::string decrypted_text = PkiEncryptor::privateKeyDecrypt(cipher_text);
  ASSERT_EQ(TEST_PLAIN_TEXT, decrypted_text);
}

TEST_F(PkiEncryptorTest, EncryptAndDecryptWithExistingCert) {
  PkiEncryptor::setKeyStorePath(EXISTING_KEY_STORE_PATH);
  const std::string cipher_text = PkiEncryptor::publicKeyEncrypt(TEST_PLAIN_TEXT);
  assertNoPlainTextExists(cipher_text);

  const std::string decrypted_text = PkiEncryptor::privateKeyDecrypt(cipher_text);
  ASSERT_EQ(TEST_PLAIN_TEXT, decrypted_text);
}

TEST_F(PkiEncryptorTest, EncryptAndDecryptWithExpiredCert) {
  PkiEncryptor::setKeyStorePath(EXPIRED_KEY_STORE_PATH);
  const std::string cipher_text = PkiEncryptor::publicKeyEncrypt(TEST_PLAIN_TEXT);
  assertNoPlainTextExists(cipher_text);

  const std::string decrypted_text = PkiEncryptor::privateKeyDecrypt(cipher_text);
  ASSERT_EQ(TEST_PLAIN_TEXT, decrypted_text);
}

TEST_F(PkiEncryptorTest, EncryptAndDecryptWithMismatchedKeyPairs) {
  PkiEncryptor::setKeyStorePath(MISMATCHED_KEYS_KEY_STORE_PATH);
  const std::string cipher_text = PkiEncryptor::publicKeyEncrypt(TEST_PLAIN_TEXT);
  assertNoPlainTextExists(cipher_text);

  assertException([&]() { PkiEncryptor::privateKeyDecrypt(cipher_text); },
                  "An error occurred when attempting to decrypt cipher text.");
}

TEST_F(PkiEncryptorTest, EncryptWithInvalidCert) {
  PkiEncryptor::setKeyStorePath(INVALID_CERT_KEY_STORE_PATH);
  assertException([&]() { PkiEncryptor::publicKeyEncrypt(TEST_PLAIN_TEXT); },
                  "An error occurred when attempting to encrypt plain text.");
}

TEST_F(PkiEncryptorTest, DecryptWithInvalidCert) {
  PkiEncryptor::setKeyStorePath(EXISTING_KEY_STORE_PATH);
  const std::string cipher_text = PkiEncryptor::publicKeyEncrypt(TEST_PLAIN_TEXT);
  assertNoPlainTextExists(cipher_text);

  PkiEncryptor::setKeyStorePath(INVALID_CERT_KEY_STORE_PATH);
  assertException([&]() { PkiEncryptor::privateKeyDecrypt(cipher_text); },
                  "An error occurred when attempting to decrypt cipher text.");
}

TEST_F(PkiEncryptorTest, EncryptSamePlainTextMultipleTimes) {
  PkiEncryptor::setKeyStorePath(EXISTING_KEY_STORE_PATH);
  const std::string cipher_text = PkiEncryptor::publicKeyEncrypt(TEST_PLAIN_TEXT);
  assertNoPlainTextExists(cipher_text);

  const std::string decrypted_text = PkiEncryptor::privateKeyDecrypt(cipher_text);
  ASSERT_EQ(TEST_PLAIN_TEXT, decrypted_text);

  const std::string cipher_text_2 = PkiEncryptor::publicKeyEncrypt(TEST_PLAIN_TEXT);
  assertNoPlainTextExists(cipher_text_2);

  const std::string decrypted_text_2 = PkiEncryptor::privateKeyDecrypt(cipher_text);
  ASSERT_EQ(TEST_PLAIN_TEXT, decrypted_text);

  ASSERT_NE(cipher_text, cipher_text_2);
}

TEST_F(PkiEncryptorTest, EncryptAndDecryptHalfSegment) {
  PkiEncryptor::setKeyStorePath(EXISTING_KEY_STORE_PATH);
  const auto two_segment_text = getRepeatedString(TEST_PLAIN_TEXT, 20);
  const auto cipher_text = PkiEncryptor::publicKeyEncrypt(two_segment_text);
  assertNoPlainTextExists(cipher_text);

  ASSERT_GT(two_segment_text.length(), PkiEncryptor::MAX_PLAIN_TEXT_SEGMENT_WIDTH);
  ASSERT_LT(two_segment_text.length(), 2 * PkiEncryptor::MAX_PLAIN_TEXT_SEGMENT_WIDTH);
  ASSERT_EQ(cipher_text.length(), 2 * PkiEncryptor::CIPHER_TEXT_SEGMENT_WIDTH);

  const auto decrypted_text = PkiEncryptor::privateKeyDecrypt(cipher_text);
  ASSERT_EQ(two_segment_text, decrypted_text);
}

TEST_F(PkiEncryptorTest, EncryptAndDecryptBigText) {
  PkiEncryptor::setKeyStorePath(EXISTING_KEY_STORE_PATH);
  const auto big_text = getRepeatedString(TEST_PLAIN_TEXT, 1000);
  const auto cipher_text = PkiEncryptor::publicKeyEncrypt(big_text);
  assertNoPlainTextExists(cipher_text);

  const auto decrypted_text = PkiEncryptor::privateKeyDecrypt(cipher_text);
  ASSERT_EQ(big_text, decrypted_text);
}

TEST_F(PkiEncryptorTest, DecryptLegacyCipherText) {
  PkiEncryptor::setKeyStorePath(EXISTING_KEY_STORE_PATH);
  assertNoPlainTextExists(LEGACY_ENCRYPTED_CIPHER_TEXT);

  const std::string decrypted_text =
      PkiEncryptor::privateKeyDecrypt(LEGACY_ENCRYPTED_CIPHER_TEXT);
  ASSERT_EQ(TEST_PLAIN_TEXT, decrypted_text);
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  return err;
}
