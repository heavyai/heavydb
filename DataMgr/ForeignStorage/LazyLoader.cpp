/*
 * Copyright 2020 OmniSci, Inc.
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

#include "LazyLoader.h"

#include <algorithm>
#include <fstream>
#include <future>

#include "Shared/mapd_glob.h"

namespace foreign_storage {

// Performs metadata scan and lazy loading operations on CSV files
CsvLazyLoader::CsvLazyLoader(Importer_NS::Loader* loader,
                             const std::string& file,
                             const Importer_NS::CopyParams& copy_params)
    : Importer_NS::Importer(loader, file, copy_params) {
  auto file_paths = mapd_glob(file);
  // These limitations will be removed before productization
  if (file_paths.size() > 1) {
    throw std::runtime_error(
        "Only single file CSV archives supported for Foreign Storage Interface.");
  }
  if (!copy_params.plain_text) {
    throw std::runtime_error(
        "Only uncompressed CSV archives supported for Foreign Storage");
  }
};

// Iterates over entire file and returns file offsets and import buffers
void CsvLazyLoader::scanMetadata() {
  auto timer = DEBUG_TIMER(__func__);
  size_t file_offset = 0;
  const auto& header_param = get_copy_params().has_header;
  if (header_param != Importer_NS::ImportHeaderRow::NO_HEADER) {
    std::ifstream file{file_path};
    CHECK(file.good());
    std::string line;
    std::getline(file, line, get_copy_params().line_delim);
    file.close();
    file_offset = line.size() + 1;
  }
  loadDelimited(file_path, true, true, file_offset);
}

// Loads given file regions
void CsvLazyLoader::fetchRegions(const FileRegions& file_regions) {
  auto timer = DEBUG_TIMER(__func__);
  auto file = fopen(file_path.c_str(), "rb");
  if (!file) {
    throw std::runtime_error{"An error occurred when attempting to open file \"" +
                             file_path + "\". " + strerror(errno)};
  }

  auto num_threads = get_copy_params().threads;
  if (num_threads == 0) {
    num_threads = std::thread::hardware_concurrency();
  }
  CHECK(num_threads);
  initializeImportBuffers(num_threads);

  std::vector<std::future<void>> futures{};
  size_t thread_index = 0;
  int batch_size = (file_regions.size() + num_threads - 1) / num_threads;
  for (size_t i = 0; i < file_regions.size(); i += batch_size) {
    futures.emplace_back(
        std::async(std::launch::async,
                   loadFileRegions,
                   file_regions,
                   i,
                   std::min<size_t>(i + batch_size - 1, file_regions.size() - 1),
                   thread_index++,
                   file,
                   std::ref(file_access_mutex_),
                   this));
  }

  for (const auto& future : futures) {
    future.wait();
  }

  fclose(file);
}

void CsvLazyLoader::loadFileRegions(const FileRegions& file_regions,
                                    const size_t start_index,
                                    const size_t end_index,
                                    const size_t thread_id,
                                    FILE* file,
                                    std::mutex& file_access_mutex,
                                    Importer_NS::Importer* importer) {
  auto timer = DEBUG_TIMER(__func__);
  const auto& buffer_size = importer->get_copy_params().buffer_size;
  for (size_t i = start_index; i <= end_index; i++) {
    CHECK(file_regions[i].region_size <= buffer_size);
    // TODO: Update importThreadDelimited to allow for reuse of file buffers
    std::unique_ptr<char[]> file_buffer = std::make_unique<char[]>(buffer_size);
    size_t read_size;
    {
      std::lock_guard<std::mutex> lock(file_access_mutex);
      fseek(file, file_regions[i].first_row_file_offset, SEEK_SET);
      read_size = fread(file_buffer.get(), 1, file_regions[i].region_size, file);
    }
    CHECK_EQ(file_regions[i].region_size, read_size);
    importThreadDelimited(thread_id,
                          importer,
                          std::move(file_buffer),
                          0,
                          file_regions[i].region_size,
                          file_regions[i].region_size,
                          {},
                          0,
                          true,
                          file_regions[i].first_row_file_offset);
  }
}
}  // namespace foreign_storage
