/*
 * Copyright 2022 HEAVY.AI, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file SessionsStoreTest.cpp
 * @brief Test suite for distributed sessions store (single node environment)
 *
 */

#include <gtest/gtest.h>
#include <boost/filesystem.hpp>
#include <string>

#include "Calcite/Calcite.h"
#include "Catalog/Catalog.h"
#include "Catalog/SessionsStore.h"
#include "Catalog/SysCatalog.h"
#include "CudaMgr/CudaMgr.h"
#include "Shared/StringTransform.h"
#include "Tests/DBHandlerTestHelpers.h"
#include "Tests/TestHelpers.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

#ifndef CALCITEPORT
#define CALCITEPORT 3279
#endif

using namespace Catalog_Namespace;
namespace fs = boost::filesystem;

namespace {

auto& sys_cat = Catalog_Namespace::SysCatalog::instance();
const std::string store_path_ = std::string(BASE_PATH) + "/sessions";

bool sessionsMatch(const SessionInfo& a, const SessionInfo& b) {
  return a.get_public_session_id() == b.get_public_session_id() &&
         a.get_session_id() == b.get_session_id() &&
         a.getCatalog().getDatabaseId() == b.getCatalog().getDatabaseId() &&
         a.get_currentUser().userId == b.get_currentUser().userId &&
         a.get_connection_info() == b.get_connection_info() &&
         a.get_executor_device_type() == b.get_executor_device_type();
}

void noopCallback(SessionInfoPtr&) {}

const std::string users[] = {"test_user_1", "test_user_2"};
const std::string dbs[] = {"test_db_1", "test_db_2"};

}  // namespace

class SessionsStoreTest : public testing::Test {
  void grantDBToUser(const std::string db_name, const std::string& user_name) {
    DBObject mapd_object(db_name, DBObjectType::DatabaseDBObjectType);
    mapd_object.setPrivileges(AccessPrivileges::ALL_DATABASE);
    auto db_cat = sys_cat.getCatalog(db_name);
    mapd_object.loadKey(*db_cat);
    sys_cat.grantDBObjectPrivileges(user_name, mapd_object, *db_cat);
  }

  void createDB(const std::string& db_name) {
    if (!sys_cat.getCatalog(db_name)) {
      sys_cat.createDatabase(db_name, shared::kRootUserId);
    }
  }

  void createUser(const std::string& user_name) {
    sys_cat.createUser(
        user_name,
        Catalog_Namespace::UserAlterations{
            "password", /*is_super=*/false, /*default_db=*/"", /*can_login=*/true},
        false);
  }

 protected:
  void SetUp() override {
    createDB("test_db_1");
    createDB("test_db_2");
    createUser("test_user_1");
    createUser("test_user_2");
    grantDBToUser("test_db_1", "test_user_1");
    grantDBToUser("test_db_1", "test_user_2");
    grantDBToUser("test_db_2", "test_user_1");
    grantDBToUser("test_db_2", "test_user_2");
  }

  void TearDown() override {
    DBMetadata db;
    sys_cat.getMetadataForDB("test_db_1", db);
    sys_cat.dropDatabase(db);
    sys_cat.getMetadataForDB("test_db_2", db);
    sys_cat.dropDatabase(db);
    sys_cat.dropUser("test_user_1");
    sys_cat.dropUser("test_user_2");
  }

 public:
  SessionInfoPtr createSession(const std::string& user, const std::string& db) const {
    auto cat = sys_cat.getCatalog(db);
    UserMetadata bob_md;
    sys_cat.getMetadataForUser(user, bob_md);
    return std::make_shared<SessionInfo>(
        cat, bob_md, ExecutorDeviceType::CPU, generate_random_string(32));
  }

  UserMetadata get_user_md(int i) {
    auto user = sys_cat.getUser(users[i]);
    CHECK(user.has_value());
    return user.value();
  }

  std::shared_ptr<Catalog> get_db_catalog(int i) {
    auto cat = sys_cat.getCatalog(dbs[i]);
    CHECK(cat);
    return cat;
  }
};

void check_session(const SessionInfoPtr& session, int id) {
  ASSERT_TRUE(session != nullptr);
  ASSERT_TRUE(session->get_currentUser().userName == users[id] &&
              session->get_catalog_ptr() == sys_cat.getCatalog(dbs[id]) &&
              session->get_executor_device_type() == ExecutorDeviceType::CPU);
}

TEST_F(SessionsStoreTest, AddGet) {
  auto store = SessionsStore::create(BASE_PATH, 1, 1, 1, -1, noopCallback);
  auto session1 = store->add(get_user_md(0), get_db_catalog(0), ExecutorDeviceType::CPU);
  auto session2 = store->add(get_user_md(1), get_db_catalog(1), ExecutorDeviceType::CPU);
  check_session(session1, 0);
  check_session(session2, 1);
  auto session_from_store = store->get(session1->get_session_id());
  ASSERT_TRUE(sessionsMatch(*session1, *session_from_store));
  session_from_store = store->get(session2->get_session_id());
  ASSERT_TRUE(sessionsMatch(*session2, *session_from_store));
}

TEST_F(SessionsStoreTest, Erase) {
  auto store = SessionsStore::create(BASE_PATH, 1, 1, 1, -1, noopCallback);
  auto session1 = store->add(get_user_md(0), get_db_catalog(0), ExecutorDeviceType::CPU);
  auto session2 = store->add(get_user_md(1), get_db_catalog(1), ExecutorDeviceType::CPU);
  check_session(session1, 0);
  check_session(session2, 1);
  store->erase(session1->get_session_id());
  ASSERT_EQ(store->get(session1->get_session_id()), nullptr);
  auto session_from_store = store->get(session2->get_session_id());
  ASSERT_TRUE(sessionsMatch(*session2, *session_from_store));
}

TEST_F(SessionsStoreTest, EraseByUser) {
  auto store = SessionsStore::create(BASE_PATH, 1, 1, 1, -1, noopCallback);
  auto session1 = store->add(get_user_md(0), get_db_catalog(0), ExecutorDeviceType::CPU);
  auto session2 = store->add(get_user_md(1), get_db_catalog(1), ExecutorDeviceType::CPU);
  check_session(session1, 0);
  check_session(session2, 1);
  store->eraseByUser(users[0]);
  ASSERT_EQ(store->get(session1->get_session_id()), nullptr);
  auto session_from_store = store->get(session2->get_session_id());
  ASSERT_TRUE(sessionsMatch(*session2, *session_from_store));
}

TEST_F(SessionsStoreTest, EraseByDB) {
  auto store = SessionsStore::create(BASE_PATH, 1, 1, 1, -1, noopCallback);
  auto session1 = store->add(get_user_md(0), get_db_catalog(0), ExecutorDeviceType::CPU);
  auto session2 = store->add(get_user_md(1), get_db_catalog(1), ExecutorDeviceType::CPU);
  check_session(session1, 0);
  check_session(session2, 1);
  store->eraseByDB(dbs[0]);
  ASSERT_EQ(store->get(session1->get_session_id()), nullptr);
  auto session_from_store = store->get(session2->get_session_id());
  ASSERT_TRUE(sessionsMatch(*session2, *session_from_store));
}

TEST_F(SessionsStoreTest, GetAll) {
  auto store = SessionsStore::create(BASE_PATH, 1, 1, 1, -1, noopCallback);
  auto session1 = store->add(get_user_md(0), get_db_catalog(0), ExecutorDeviceType::CPU);
  auto session2 = store->add(get_user_md(1), get_db_catalog(1), ExecutorDeviceType::CPU);
  check_session(session1, 0);
  check_session(session2, 1);
  auto all_sessions = store->getAllSessions();
  ASSERT_EQ(all_sessions.size(), 2ul);
  bool match = (sessionsMatch(*all_sessions[0], *session1) &&
                sessionsMatch(*all_sessions[1], *session2)) ||
               (sessionsMatch(*all_sessions[0], *session2) &&
                sessionsMatch(*all_sessions[1], *session1));
  ASSERT_TRUE(match);
}

TEST_F(SessionsStoreTest, GetUserSessions) {
  auto store = SessionsStore::create(BASE_PATH, 1, 1, 1, -1, noopCallback);
  auto session1 = store->add(get_user_md(0), get_db_catalog(0), ExecutorDeviceType::CPU);
  auto session2 = store->add(get_user_md(1), get_db_catalog(1), ExecutorDeviceType::CPU);
  check_session(session1, 0);
  check_session(session2, 1);
  auto user_sessions = store->getUserSessions(users[0]);
  ASSERT_EQ(user_sessions.size(), 1ul);
  ASSERT_TRUE(sessionsMatch(*session1, *user_sessions[0]));
}

TEST_F(SessionsStoreTest, GetByPublicID) {
  auto store = SessionsStore::create(BASE_PATH, 1, 1, 1, -1, noopCallback);
  auto session1 = store->add(get_user_md(0), get_db_catalog(0), ExecutorDeviceType::CPU);
  auto session2 = store->add(get_user_md(1), get_db_catalog(1), ExecutorDeviceType::CPU);
  check_session(session1, 0);
  check_session(session2, 1);
  ASSERT_TRUE(
      sessionsMatch(*session1, *store->getByPublicID(session1->get_public_session_id())));
}

TEST_F(SessionsStoreTest, DisconnectTest) {
  bool disconnected = false;
  auto store = SessionsStore::create(
      BASE_PATH, 1, 1, 1, -1, [&disconnected](SessionInfoPtr&) { disconnected = true; });
  auto session1 = store->add(get_user_md(0), get_db_catalog(0), ExecutorDeviceType::CPU);
  auto session2 = store->add(get_user_md(1), get_db_catalog(1), ExecutorDeviceType::CPU);
  check_session(session1, 0);
  check_session(session2, 1);
  store->disconnect(session1->get_session_id());
  ASSERT_TRUE(store->get(session1->get_session_id()) == nullptr);
  ASSERT_TRUE(store->get(session2->get_session_id()) != nullptr);
  ASSERT_TRUE(disconnected);
}

TEST_F(SessionsStoreTest, GetCopy) {
  auto store = SessionsStore::create(BASE_PATH, 1, 1, 1, -1, noopCallback);
  auto session1 = store->add(get_user_md(0), get_db_catalog(0), ExecutorDeviceType::CPU);
  auto session2 = store->add(get_user_md(1), get_db_catalog(1), ExecutorDeviceType::CPU);
  check_session(session1, 0);
  check_session(session2, 1);
  ASSERT_TRUE(
      sessionsMatch(*session1, store->getSessionCopy(session1->get_session_id())));
  ASSERT_TRUE(
      sessionsMatch(*session2, store->getSessionCopy(session2->get_session_id())));
}

TEST_F(SessionsStoreTest, StoreOverflow) {
  auto store = SessionsStore::create(BASE_PATH, 1, 1, 1, 2, noopCallback);
  auto session1 = store->add(get_user_md(0), get_db_catalog(0), ExecutorDeviceType::CPU);
  auto session2 = store->add(get_user_md(1), get_db_catalog(1), ExecutorDeviceType::CPU);
  check_session(session1, 0);
  check_session(session2, 1);
  try {
    store->add(get_user_md(0), get_db_catalog(0), ExecutorDeviceType::CPU);
    ASSERT_TRUE(false) << "No exception thrown";
  } catch (const std::exception& e) {
    ASSERT_EQ(std::string(e.what()), "Too many active sessions");
  }
}

// For a session to be expired we need to nullify all shared_pointers pointing to it
TEST_F(SessionsStoreTest, GetWithTimeout) {
  auto store = SessionsStore::create(BASE_PATH, 1, 1, 1, -1, noopCallback);
  auto session1 = store->add(get_user_md(0), get_db_catalog(0), ExecutorDeviceType::CPU);
  check_session(session1, 0);
  sleep(2);
  auto session_id = session1->get_session_id();
  session1 = nullptr;
  ASSERT_EQ(store->get(session_id), nullptr);
}

// This test case:
// 1) creates a store with limited capacity of two sessions
// 2) creates two sessions
// 3) expires the first one
// 4) creates another two sessions - one successfuly, another fails with store overflow
TEST_F(SessionsStoreTest, StoreOverflowExpired) {
  auto store = SessionsStore::create(BASE_PATH, 1, 3, 3, 2, noopCallback);
  auto session1 = store->add(get_user_md(0), get_db_catalog(0), ExecutorDeviceType::CPU);
  check_session(session1, 0);
  auto session1_id = session1->get_session_id();
  session1 = nullptr;
  sleep(2);
  auto session2 = store->add(get_user_md(1), get_db_catalog(1), ExecutorDeviceType::CPU);
  check_session(session2, 1);
  ASSERT_EQ(store->getAllSessions().size(), 2ul);
  auto session2_id = session2->get_session_id();
  session2 = nullptr;
  sleep(2);
  auto session3 = store->add(get_user_md(0), get_db_catalog(0), ExecutorDeviceType::CPU);
  auto session3_id = session3->get_session_id();
  ASSERT_TRUE(session3 != nullptr);
  session3 = nullptr;
  try {
    store->add(get_user_md(0), get_db_catalog(0), ExecutorDeviceType::CPU);
    ASSERT_TRUE(false) << "No exception thrown";
  } catch (const std::exception& e) {
    ASSERT_EQ(std::string(e.what()), "Too many active sessions");
  }
  ASSERT_EQ(store->getAllSessions().size(), 2ul);
  ASSERT_TRUE(store->get(session1_id) == nullptr);
  ASSERT_TRUE(store->get(session2_id) != nullptr);
  ASSERT_TRUE(store->get(session3_id) != nullptr);
}

TEST_F(SessionsStoreTest, NoTimeoutWhenInUse) {
  auto store = SessionsStore::create(BASE_PATH, 1, 1, 1, -1, noopCallback);
  auto session1 = store->add(get_user_md(0), get_db_catalog(0), ExecutorDeviceType::CPU);
  check_session(session1, 0);
  // we sleep longer than max_idle_time, but we still have the shared_ptr to the session
  // so it should not be expired, because it is in "active" use.
  sleep(2);
  ASSERT_TRUE(sessionsMatch(*store->get(session1->get_session_id()), *session1));
}

// Test for a very simple concurrent sessions lifecycle:
// 1) creates #n_workers threads
// 2) each thread runs multiple session lifeccle:
//    - create session
//    - read a few times
//    - erase 50% of them
// 3) after all threads are done wait couple seconds for the rest of the sessions
//    to time out
// 4) check if there are no sessions left in the store
TEST_F(SessionsStoreTest, ConcurrentSessionLifeCycle) {
  const int n_sessions = 1024;
  std::atomic<int> current_session = 0;
  std::vector<std::string> session_ids(n_sessions);
  auto store = SessionsStore::create(BASE_PATH, 1, 1, 1, -1, noopCallback);
  auto session_work = [&](int i) {
    while (true) {
      auto session_id_pos = current_session++;
      if (session_id_pos >= n_sessions) {
        return;
      }
      auto session =
          store->add(get_user_md(0), get_db_catalog(0), ExecutorDeviceType::CPU);
      ASSERT_TRUE(session != nullptr);
      const auto& session_id = session->get_session_id();
      session_ids[session_id_pos] = session_id;
      for (int j = 0; j < 10; ++j) {
        ASSERT_TRUE(sessionsMatch(*session, *store->get(session_id)));
      }
      if (i % 2) {
        store->erase(session_id);
        ASSERT_EQ(store->get(session_id), nullptr);
      }
    }
  };
  int n_workers = 16;
  std::vector<std::thread> threads;
  threads.reserve(n_workers);
  for (int i = 0; i < n_workers; ++i) {
    threads.emplace_back(session_work, i);
  }
  for (auto& t : threads) {
    t.join();
  }
  sleep(2);
  for (const auto& session_id : session_ids) {
    ASSERT_EQ(store->get(session_id), nullptr);
  }
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  int err{0};
  SystemParameters sys_parms;
  std::string data_path = std::string(BASE_PATH) + "/" + shared::kDataDirectoryName;
  auto dummy =
      std::make_shared<Data_Namespace::DataMgr>(data_path, sys_parms, nullptr, false, 0);
  auto calcite = std::make_shared<Calcite>(
      -1, CALCITEPORT, std::string(BASE_PATH), 1024, 5000, true, "");
  sys_cat.init(BASE_PATH, dummy, {}, calcite, false, false, {}, {});
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  return err;
}
