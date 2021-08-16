// Template test file for inclusion into ThreadingTest.cpp only

TEST(THREADING, MODED(ParallelFor_)) {
    using namespace MODED(threading_);
    g_counter = 0;
    parallel_for(1, 100, [&](int i) { g_counter++; } );
    ASSERT_EQ(g_counter, 99); g_counter = 0;
    parallel_for(blocked_range<size_t>(1, 10000), [&](auto r) {
      g_counter += r.size();
    } );
    ASSERT_EQ(g_counter, 9999); g_counter = 0;
}

TEST(THREADING, MODED(ParallelReduce_)) {
    using namespace MODED(threading_);
    g_counter = 0;
    int res = parallel_reduce(blocked_range<size_t>(1, 10000), int(0), [&](auto r, int v) {
      g_counter += r.size();
      return int(v + r.size());
    }, std::plus<int>()  );
    ASSERT_EQ(g_counter, res); g_counter = 0;
}

TEST(THREADING, MODED(Async_)) {
    using namespace MODED(threading_);
    g_counter = 0;
    auto a1 = async([&](int i){ g_counter++; ASSERT_EQ(i, 1); }, 1);
    a1.wait(); ASSERT_EQ( g_counter, 1 );
}

TEST(THREADING, MODED(TaskGroup_)) {
    using namespace MODED(threading_);
    g_counter = 0;
    task_group tg;
    tg.run([&]{ g_counter++; });
    tg.wait(); ASSERT_EQ( g_counter, 1 );
}
