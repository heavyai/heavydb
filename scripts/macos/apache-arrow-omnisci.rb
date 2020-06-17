class ApacheArrowOmnisci < Formula
  desc "Columnar in-memory analytics layer designed to accelerate big data"
  homepage "https://arrow.apache.org/"
  url "https://github.com/apache/arrow/archive/apache-arrow-0.16.0.tar.gz"
  head "https://github.com/apache/arrow.git"
  sha256 "d7b3838758a365c8c47d55ab0df1006a70db951c6964440ba354f81f518b8d8d"
  revision 2

  depends_on "cmake" => :build
  depends_on "boost"

  patch :DATA

  def install
    ENV.cxx11
    args = [
      "-DCMAKE_BUILD_TYPE=Release",
      "-DCMAKE_INSTALL_PREFIX=#{prefix}",
      "-DARROW_BUILD_SHARED=ON",
      "-DARROW_BUILD_STATIC=ON",
      "-DARROW_BUILD_TESTS=OFF",
      "-DARROW_BUILD_BENCHMARKS=OFF",
      "-DARROW_WITH_BROTLI=OFF",
      "-DARROW_WITH_ZLIB=ON",
      "-DARROW_USE_GLOG=OFF",
      "-DARROW_GFLAGS_USE_SHARED=OFF",
      "-DARROW_WITH_LZ4=OFF",
      "-DARROW_WITH_SNAPPY=ON",
      "-DARROW_WITH_ZSTD=ON",
      "-DARROW_CSV=ON",
      "-DARROW_JSON=ON",
      "-DARROW_BOOST_USE_SHARED=ON",
      "-DARROW_PARQUET=ON",
      "-DARROW_JEMALLOC=OFF",
      "-DTHRIFT_HOME=#{Formula["thrift"].opt_prefix}"
    ]

    cd "cpp" do
      system "cmake", ".", *std_cmake_args, *args
      system "make"
      system "make", "install"
    end
  end

  test do
    (testpath/"test.cpp").write <<~EOS
      #include "arrow/api.h"
      int main(void)
      {
        arrow::int64();
        return 0;
      }
    EOS
    system ENV.cxx, "test.cpp", "-std=c++11", "-I#{include}", "-L#{lib}", "-larrow", "-o", "test"
    system "./test"
  end
end

__END__
diff --git a/cpp/src/parquet/arrow/arrow_reader_writer_test.cc b/cpp/src/parquet/arrow/arrow_reader_writer_test.cc
index b47d9334d..68db64469 100644
--- a/cpp/src/parquet/arrow/arrow_reader_writer_test.cc
+++ b/cpp/src/parquet/arrow/arrow_reader_writer_test.cc
@@ -1931,6 +1931,54 @@ TEST(TestArrowReadWrite, ReadSingleRowGroup) {
   AssertTablesEqual(*table, *concatenated, /*same_chunk_layout=*/false);
 }
 
+//  Exercise reading table manually with nested RowGroup and Column loops, i.e.
+//
+//  for (int i = 0; i < n_row_groups; i++)
+//    for (int j = 0; j < n_cols; j++)
+//      reader->RowGroup(i)->Column(j)->Read(&chunked_array);
+::arrow::Result<std::shared_ptr<Table>> ReadTableManually(FileReader* reader) {
+  std::vector<std::shared_ptr<Table>> tables;
+
+  std::shared_ptr<::arrow::Schema> schema;
+  RETURN_NOT_OK(reader->GetSchema(&schema));
+
+  int n_row_groups = reader->num_row_groups();
+  int n_columns = schema->num_fields();
+  for (int i = 0; i < n_row_groups; i++) {
+    std::vector<std::shared_ptr<ChunkedArray>> columns{static_cast<size_t>(n_columns)};
+
+    for (int j = 0; j < n_columns; j++) {
+      RETURN_NOT_OK(reader->RowGroup(i)->Column(j)->Read(&columns[j]));
+    }
+
+    tables.push_back(Table::Make(schema, columns));
+  }
+
+  return ConcatenateTables(tables);
+}
+
+TEST(TestArrowReadWrite, ReadTableManually) {
+  const int num_columns = 1;
+  const int num_rows = 128;
+
+  std::shared_ptr<Table> expected;
+  ASSERT_NO_FATAL_FAILURE(MakeDoubleTable(num_columns, num_rows, 1, &expected));
+
+  std::shared_ptr<Buffer> buffer;
+  ASSERT_NO_FATAL_FAILURE(WriteTableToBuffer(expected, num_rows / 2,
+                                             default_arrow_writer_properties(), &buffer));
+
+  std::unique_ptr<FileReader> reader;
+  ASSERT_OK_NO_THROW(OpenFile(std::make_shared<BufferReader>(buffer),
+                              ::arrow::default_memory_pool(), &reader));
+
+  ASSERT_EQ(2, reader->num_row_groups());
+
+  ASSERT_OK_AND_ASSIGN(auto actual, ReadTableManually(reader.get()));
+
+  AssertTablesEqual(*actual, *expected, /*same_chunk_layout=*/false);
+}
+
 TEST(TestArrowReadWrite, GetRecordBatchReader) {
   const int num_columns = 20;
   const int num_rows = 1000;
diff --git a/cpp/src/parquet/arrow/reader.cc b/cpp/src/parquet/arrow/reader.cc
index f75aca895..b341c6206 100644
--- a/cpp/src/parquet/arrow/reader.cc
+++ b/cpp/src/parquet/arrow/reader.cc
@@ -783,7 +783,7 @@ Status FileReaderImpl::GetColumn(int i, FileColumnIteratorFactory iterator_facto
   auto ctx = std::make_shared<ReaderContext>();
   ctx->reader = reader_.get();
   ctx->pool = pool_;
-  ctx->iterator_factory = AllRowGroupsFactory();
+  ctx->iterator_factory = iterator_factory;
   ctx->filter_leaves = false;
   std::unique_ptr<ColumnReaderImpl> result;
   RETURN_NOT_OK(GetReader(manifest_.schema_fields[i], ctx, &result));
-- 
2.26.2

