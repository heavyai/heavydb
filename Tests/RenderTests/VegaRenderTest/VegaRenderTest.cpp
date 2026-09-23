/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <Catalog/Catalog.h>
#include <QueryEngine/Rendering/RenderInfo.h>
#include <QueryEngine/ResultSet.h>
#include <QueryRenderer/Interface/RenderQueryExecuteData.h>
#include <QueryRenderer/Interface/RenderQueryRunnerInterface.h>
#include <QueryRenderer/Interop/InteropBufferHandle.h>
#include <QueryRenderer/QueryRenderManager.h>
#include <QueryRenderer/Utils/RapidJSONUtils.h>
#include <QueryRenderer/Utils/StringUtils.h>

#ifdef HAVE_CUDA
#include "CudaMgr/CudaMgr.h"
#include "VegaRenderTestCudaSetup.h"
#else  // !HAVE_CUDA
#include "VegaRenderTestCpuSetup.h"
#endif  // HAVE_CUDA

#include <Logger/Logger.h>
#include <Shared/scope.h>

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>

#include <boost/algorithm/string/join.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <fstream>
#include <vector>

using namespace QueryRenderer;

inline void throw_cuda_json_parse_error(const std::string& errstr) {
  throw std::runtime_error("JSON parse error in cuda details: " + errstr);
}

struct CudaRuntimeDetails {
  int startGpu;
  int numGpus;
  std::set<size_t> gpus;

  CudaRuntimeDetails(QueryRenderManager& renderMgr, const rapidjson::Value& obj)
      : startGpu(0), numGpus(-1), renderMgr_(renderMgr) {
    rapidjson::Value::ConstMemberIterator mitr;

    if ((mitr = obj.FindMember("startGpu")) != obj.MemberEnd()) {
      if (!mitr->value.IsInt()) {
        throw_cuda_json_parse_error("\"startGpu\" property must be an integer.");
      }
      startGpu = mitr->value.GetInt();
    }

    if ((mitr = obj.FindMember("numGpus")) != obj.MemberEnd()) {
      if (!mitr->value.IsInt()) {
        throw_cuda_json_parse_error("\"numGpus\" property must be an integer.");
      }
      numGpus = mitr->value.GetInt();
    }

    if ((mitr = obj.FindMember("gpus")) != obj.MemberEnd()) {
      if (!mitr->value.IsArray()) {
        throw_cuda_json_parse_error("\"gpus\" property must be an array.");
      }

      for (rapidjson::Value::ConstValueIterator itr = mitr->value.Begin();
           itr != mitr->value.End();
           ++itr) {
        if (!itr->IsInt()) {
          throw_cuda_json_parse_error(
              "All elements of the \"gpus\" property must be integers.");
        }
        gpus.insert(itr->GetInt());
      }
    }
  }
  virtual ~CudaRuntimeDetails() {}
  virtual size_t getUsedBytes() const = 0;

  virtual void initExecute() {}
  virtual void preExecuteGpu(const int, const size_t) {}
  virtual void executeGpu(const int, const size_t) {}
  virtual void cleanupExecuteGpu(const int, const bool, const size_t) {}

  virtual QueryDataLayoutShPtr getVboLayout() const { return nullptr; }
  virtual QueryDataLayoutShPtr getUboLayout() const { return nullptr; }

 protected:
  QueryRenderManager& renderMgr_;
};

#ifdef HAVE_CUDA
struct SimplePointRowRuntimeDetails : public CudaRuntimeDetails {
  int numPts;
  int seed;
  std::array<double, 2> xExtents;
  std::array<double, 2> yExtents;
  std::array<double, 2> valExtents;
  std::array<int64_t, 2> partyExtents;

  SimplePointRowRuntimeDetails(QueryRenderManager& renderMgr, const rapidjson::Value& obj)
      : CudaRuntimeDetails(renderMgr, obj)
      , numPts(100)
      , seed(1234)
      , seedDiff(500)
      , runtimeSeedToUse(0) {
    rapidjson::Value::ConstMemberIterator mitr;
    if ((mitr = obj.FindMember("x")) == obj.MemberEnd() || !mitr->value.IsArray() ||
        mitr->value.Size() != 2) {
      throw std::runtime_error(
          "SimplePointType must have an \"x\" property and it must be a numeric array of "
          "size 2");
    }
    rapidjson::Value::ConstValueIterator itr;
    int idx = 0;
    for (itr = mitr->value.Begin(); itr != mitr->value.End(); ++itr, ++idx) {
      if (!itr->IsNumber()) {
        throw std::runtime_error("All SimplePointType \"x\" values must be doubles. " +
                                 RapidJSONUtils::getObjAsString(*itr) +
                                 " is not a double.");
      }
      xExtents[idx] = itr->GetDouble();
    }

    if ((mitr = obj.FindMember("y")) == obj.MemberEnd() || !mitr->value.IsArray() ||
        mitr->value.Size() != 2) {
      throw std::runtime_error(
          "SimplePointType must have an \"y\" property and it must be a numeric array of "
          "size 2");
    }
    for (itr = mitr->value.Begin(), idx = 0; itr != mitr->value.End(); ++itr, ++idx) {
      if (!itr->IsNumber()) {
        throw std::runtime_error("All SimplePointType \"y\" values must be doubles. " +
                                 RapidJSONUtils::getObjAsString(*itr) +
                                 " is not a double.");
      }
      yExtents[idx] = itr->GetDouble();
    }

    if ((mitr = obj.FindMember("val")) == obj.MemberEnd() || !mitr->value.IsArray() ||
        mitr->value.Size() != 2) {
      throw std::runtime_error(
          "SimplePointType must have an \"val\" property and it must be a numeric array "
          "of size 2");
    }
    for (itr = mitr->value.Begin(), idx = 0; itr != mitr->value.End(); ++itr, ++idx) {
      if (!itr->IsNumber()) {
        throw std::runtime_error("All SimplePointType \"val\" values must be doubles. " +
                                 RapidJSONUtils::getObjAsString(*itr) +
                                 " is not a double.");
      }
      valExtents[idx] = itr->GetDouble();
    }

    if ((mitr = obj.FindMember("party")) == obj.MemberEnd() || !mitr->value.IsArray() ||
        mitr->value.Size() != 2) {
      throw std::runtime_error(
          "SimplePointType must have an \"party\" property and it must be a numeric "
          "array of size 2");
    }
    for (itr = mitr->value.Begin(), idx = 0; itr != mitr->value.End(); ++itr, ++idx) {
      if (!itr->IsInt64()) {
        throw std::runtime_error(
            "All SimplePointType \"party\" values must be 64-bit ints. " +
            RapidJSONUtils::getObjAsString(*itr) + " is not an int64.");
      }
      partyExtents[idx] = itr->GetInt64();
    }

    if ((mitr = obj.FindMember("num")) != obj.MemberEnd()) {
      if (!mitr->value.IsInt()) {
        throw_cuda_json_parse_error("\"num\" property must be an integer.");
      }
      numPts = mitr->value.GetInt();
    }

    if ((mitr = obj.FindMember("randseed")) != obj.MemberEnd()) {
      if (!mitr->value.IsInt()) {
        throw_cuda_json_parse_error("\"randseed\" property must be an integer.");
      }
      seed = mitr->value.GetInt();
    }

    runtimeSeedToUse = seed + startGpu * seedDiff;
    queryDataLayoutPtr = SimplePointRow::getQueryDataLayout();
  }

  size_t getUsedBytes() const final { return numPts * sizeof(SimplePointRow); }

  void preExecuteGpu(const int gpuIdx, const size_t usedBytes) final {
    cudaHandle = renderMgr_.getQueryOutputBufferDescriptor(gpuIdx);
    if (usedBytes > cudaHandle.num_bytes) {
      throw std::runtime_error(
          "Attempting to generate " + std::to_string(usedBytes) +
          " bytes of data but the buffer has only been allocated for " +
          std::to_string(cudaHandle.num_bytes) + " bytes.");
    }
  }

  void executeGpu(const int gpuIdx, const size_t usedBytes) final {
    curandState* d_states;
    cudaMalloc((void**)&d_states, sizeof(curandState) * numPts);
    ScopeGuard release = [&d_states]() { cudaFree(d_states); };

    SimplePointRow::setup_kernel(d_states, numPts, 1, runtimeSeedToUse);

    cudaMemset(cudaHandle.handle, 0, usedBytes);

    SimplePointRow::get_random_data(d_states,
                                    (SimplePointRow*)cudaHandle.handle,
                                    numPts,
                                    xExtents,
                                    yExtents,
                                    valExtents,
                                    partyExtents,
                                    numPts,
                                    1);

    runtimeSeedToUse += seedDiff;
  }

  void cleanupExecuteGpu(const int gpuIdx,
                         const bool gpuUsed,
                         const size_t usedBytes) final {
    renderMgr_.releaseQueryOutputBufferDescriptor(
        gpuIdx, (gpuUsed ? usedBytes : 0), queryDataLayoutPtr);
  }

  QueryDataLayoutShPtr getVboLayout() const final { return queryDataLayoutPtr; }

 private:
  QueryDataLayoutShPtr queryDataLayoutPtr;
  BufferMemoryDescriptor cudaHandle;
  const int seedDiff;
  int runtimeSeedToUse;
};

using CreateRuntimeDetailsFunc =
    std::function<std::shared_ptr<CudaRuntimeDetails>(QueryRenderManager&,
                                                      const rapidjson::Value&)>;

std::pair<QueryDataLayoutShPtr, QueryDataLayoutShPtr> runCuda(
    QueryRenderManager& renderManager,
    const std::string& cudaDetails) {
  rapidjson::Document json;
  json.Parse(cudaDetails.c_str());
  if (json.HasParseError()) {
    throw_cuda_json_parse_error(
        "offset: " + std::to_string(json.GetErrorOffset()) +
        ", error msg: " + rapidjson::GetParseError_En(json.GetParseError()));
  }

  if (!json.IsObject()) {
    throw_cuda_json_parse_error("Cuda details must be a json object.");
  }

  rapidjson::Value::ConstMemberIterator mitr;
  if ((mitr = json.FindMember("type")) == json.MemberEnd() || !mitr->value.IsString()) {
    throw_cuda_json_parse_error(
        "Cuda details object must contain a \"type\" and it must be a string.");
  }

  auto typestr = makeLowerCase(std::string(mitr->value.GetString()));
  auto typelowerstr = makeLowerCase(typestr);

  CreateRuntimeDetailsFunc createRuntimeDetailsFunc;
  std::function<void(const std::shared_ptr<CudaRuntimeDetails>& details,
                     const int,
                     const bool,
                     const size_t)>
      postExecuteGpuFunc;
  if (makeLowerCase(typestr) == "simplepointrow") {
    createRuntimeDetailsFunc = [](QueryRenderManager& renderMgr,
                                  const rapidjson::Value& obj) {
      return std::make_shared<SimplePointRowRuntimeDetails>(renderMgr, obj);
    };
  } else {
    throw_cuda_json_parse_error("\"" + typestr + "\" is not a valid type.");
  }

  auto cuda_run_details = createRuntimeDetailsFunc(renderManager, json);
  const auto& startGpu = cuda_run_details->startGpu;
  auto& numGpus = cuda_run_details->numGpus;
  const auto& gpus = cuda_run_details->gpus;
  size_t bytesToUse = cuda_run_details->getUsedBytes();
  std::set<size_t> gpusToUse = gpus;
  auto cudaMgr = renderManager.getCudaMgr();
  CHECK(cudaMgr);
  auto maxNumGpus = cudaMgr->getDeviceCount();
  if (gpusToUse.size() == 0) {
    if (numGpus < 0) {
      numGpus = maxNumGpus - startGpu;
    }

    auto endGpu = startGpu + numGpus;
    CHECK(endGpu <= maxNumGpus);
    for (int i = startGpu; i < endGpu; i++) {
      gpusToUse.insert(size_t(i));
    }
  }

  cuda_run_details->initExecute();
  for (int i = 0; i < maxNumGpus; ++i) {
    bool gpuUsed = gpusToUse.find(size_t(i)) != gpusToUse.end();
    ScopeGuard loopFinalize = [&cuda_run_details, i, gpuUsed, bytesToUse]() {
      cuda_run_details->cleanupExecuteGpu(i, gpuUsed, bytesToUse);
    };

    if (gpuUsed) {
      cuda_run_details->preExecuteGpu(i, bytesToUse);
      cuda_run_details->executeGpu(i, bytesToUse);
    }
  }

  return std::make_pair(cuda_run_details->getVboLayout(),
                        cuda_run_details->getUboLayout());
}
#else  // !HAVE_CUDA

struct SimplePointRowCPURuntimeDetails : public CudaRuntimeDetails {
  int numPts;
  int seed;
  std::array<double, 2> xExtents;
  std::array<double, 2> yExtents;
  std::array<double, 2> valExtents;
  std::array<int64_t, 2> partyExtents;

  SimplePointRowCPURuntimeDetails(QueryRenderManager& renderMgr,
                                  const rapidjson::Value& obj)
      : CudaRuntimeDetails(renderMgr, obj)
      , numPts(100)
      , seed(1234)
      , seedDiff(500)
      , runtimeSeedToUse(0) {
    rapidjson::Value::ConstMemberIterator mitr;
    if ((mitr = obj.FindMember("x")) == obj.MemberEnd() || !mitr->value.IsArray() ||
        mitr->value.Size() != 2) {
      throw std::runtime_error(
          "SimplePointType must have an \"x\" property and it must be a numeric array of "
          "size 2");
    }
    rapidjson::Value::ConstValueIterator itr;
    int idx = 0;
    for (itr = mitr->value.Begin(); itr != mitr->value.End(); ++itr, ++idx) {
      if (!itr->IsNumber()) {
        throw std::runtime_error("All SimplePointType \"x\" values must be doubles. " +
                                 RapidJSONUtils::getObjAsString(*itr) +
                                 " is not a double.");
      }
      xExtents[idx] = itr->GetDouble();
    }

    if ((mitr = obj.FindMember("y")) == obj.MemberEnd() || !mitr->value.IsArray() ||
        mitr->value.Size() != 2) {
      throw std::runtime_error(
          "SimplePointType must have an \"y\" property and it must be a numeric array of "
          "size 2");
    }
    for (itr = mitr->value.Begin(), idx = 0; itr != mitr->value.End(); ++itr, ++idx) {
      if (!itr->IsNumber()) {
        throw std::runtime_error("All SimplePointType \"y\" values must be doubles. " +
                                 RapidJSONUtils::getObjAsString(*itr) +
                                 " is not a double.");
      }
      yExtents[idx] = itr->GetDouble();
    }

    if ((mitr = obj.FindMember("val")) == obj.MemberEnd() || !mitr->value.IsArray() ||
        mitr->value.Size() != 2) {
      throw std::runtime_error(
          "SimplePointType must have an \"val\" property and it must be a numeric array "
          "of size 2");
    }
    for (itr = mitr->value.Begin(), idx = 0; itr != mitr->value.End(); ++itr, ++idx) {
      if (!itr->IsNumber()) {
        throw std::runtime_error("All SimplePointType \"val\" values must be doubles. " +
                                 RapidJSONUtils::getObjAsString(*itr) +
                                 " is not a double.");
      }
      valExtents[idx] = itr->GetDouble();
    }

    if ((mitr = obj.FindMember("party")) == obj.MemberEnd() || !mitr->value.IsArray() ||
        mitr->value.Size() != 2) {
      throw std::runtime_error(
          "SimplePointType must have an \"party\" property and it must be a numeric "
          "array of size 2");
    }
    for (itr = mitr->value.Begin(), idx = 0; itr != mitr->value.End(); ++itr, ++idx) {
      if (!itr->IsInt64()) {
        throw std::runtime_error(
            "All SimplePointType \"party\" values must be 64-bit ints. " +
            RapidJSONUtils::getObjAsString(*itr) + " is not an int64.");
      }
      partyExtents[idx] = itr->GetInt64();
    }

    if ((mitr = obj.FindMember("num")) != obj.MemberEnd()) {
      if (!mitr->value.IsInt()) {
        throw_cuda_json_parse_error("\"num\" property must be an integer.");
      }
      numPts = mitr->value.GetInt();
    }

    if ((mitr = obj.FindMember("randseed")) != obj.MemberEnd()) {
      if (!mitr->value.IsInt()) {
        throw_cuda_json_parse_error("\"randseed\" property must be an integer.");
      }
      seed = mitr->value.GetInt();
    }

    runtimeSeedToUse = seed + startGpu * seedDiff;
    queryDataLayoutPtr = SimplePointRow::getQueryDataLayout();
  }

  size_t getUsedBytes() const final { return numPts * sizeof(SimplePointRow); }

  void preExecuteGpu(const int i, const size_t usedBytes) final {}

  void executeGpu(const int i, const size_t usedBytes) final {
    UniformRandomNumberGenerator<double> rand(0, 1);

    SimplePointRow* rows = new SimplePointRow[numPts];

    SimplePointRow::get_random_data(
        rand, rows, numPts, xExtents, yExtents, valExtents, partyExtents);

    renderMgr_.bufferVboData(reinterpret_cast<int8_t*>(rows), usedBytes, 0, i);

    delete[] rows;

    renderMgr_.setRenderBufferDataLayout(i, 0, usedBytes, getVboLayout());
  }

  void cleanupExecuteGpu(const int i, const bool used, const size_t usedBytes) {}

  QueryDataLayoutShPtr getVboLayout() const final { return queryDataLayoutPtr; }

 private:
  QueryDataLayoutShPtr queryDataLayoutPtr;
  const int seedDiff;
  int runtimeSeedToUse;
};

typedef std::function<std::shared_ptr<CudaRuntimeDetails>(QueryRenderManager&,
                                                          const rapidjson::Value&)>
    CreateRuntimeDetailsFunc;

std::pair<QueryDataLayoutShPtr, QueryDataLayoutShPtr> runCPUQuery(
    QueryRenderManager& renderManager,
    const std::string& queryDetails) {
  rapidjson::Document json;
  json.Parse(queryDetails.c_str());
  if (json.HasParseError()) {
    throw_cuda_json_parse_error(
        "offset: " + std::to_string(json.GetErrorOffset()) +
        ", error msg: " + rapidjson::GetParseError_En(json.GetParseError()));
  }

  if (!json.IsObject()) {
    throw_cuda_json_parse_error("Query details must be a json object.");
  }

  rapidjson::Value::ConstMemberIterator mitr;
  if ((mitr = json.FindMember("type")) == json.MemberEnd() || !mitr->value.IsString()) {
    throw_cuda_json_parse_error(
        "Query details object must contain a \"type\" and it must be a string.");
  }

  auto typestr = makeLowerCase(std::string(mitr->value.GetString()));
  auto typelowerstr = makeLowerCase(typestr);

  CreateRuntimeDetailsFunc createRuntimeDetailsFunc;
  std::function<void(const std::shared_ptr<CudaRuntimeDetails>& details,
                     const int,
                     const bool,
                     const size_t)>
      postExecuteGpuFunc;
  if (makeLowerCase(typestr) == "simplepointrow") {
    createRuntimeDetailsFunc = [](QueryRenderManager& renderMgr,
                                  const rapidjson::Value& obj) {
      return std::make_shared<SimplePointRowCPURuntimeDetails>(renderMgr, obj);
    };
  } else {
    throw_cuda_json_parse_error("\"" + typestr + "\" is not a valid type.");
  }

  auto cuda_run_details = createRuntimeDetailsFunc(renderManager, json);
  cuda_run_details->initExecute();  // TODO(adb): seems to do nothing

  size_t bytesToUse = cuda_run_details->getUsedBytes();

  cuda_run_details->preExecuteGpu(0, bytesToUse);
  cuda_run_details->executeGpu(0, bytesToUse);
  cuda_run_details->cleanupExecuteGpu(0, true, bytesToUse);

  return std::make_pair(cuda_run_details->getVboLayout(),
                        cuda_run_details->getUboLayout());
}

#endif  // HAVE_CUDA

class RenderQueryRunner : public RenderQueryRunnerInterface {
 public:
  RenderQueryRunner() = delete;
  RenderQueryRunner(const RenderSessionKey& render_session_key,
                    QueryRenderManager& render_manager)
      : render_manager_(render_manager)
      , render_info_(render_session_key,
                     RenderQueryOptions(),
                     heavyai::InSituFlags::kInSitu) {}

  void notifyQueryExecutionComplete() const final {
    // no-op. Interop is handled above.
    return;
  }

  RenderQueryParseData executeQueryParse(RenderQueryExecuteTimer&,
                                         const std::string&,
                                         const JSONLocation*,
                                         const RenderQueryOptions&,
                                         const RenderQuerySpecialtyType) final {
    // no-op for now.
    // TODO(croot): we should be careful that execution is not dependent on validation for
    // cases like this.
    CHECK(false);
    return {RenderQueryStructure({}, {}, heavyai::InSituFlags::kInSitu),
            RenderQueryBufferLayouts()};
  }

  RenderQueryExecuteData executeQuery(RenderQueryExecuteTimer& render_timer,
                                      const std::string& query_str,
                                      const JSONLocation* data_loc,
                                      const RenderQueryOptions& render_query_opts,
                                      const RenderQuerySpecialtyType render_query_type,
                                      const heavyai::InSituFlags insitu_flags) final {
    QueryDataLayoutShPtr vbolayout;
    QueryDataLayoutShPtr ubolayout;
    switch (render_query_type) {
      case RenderQuerySpecialtyType::kPolys:
      case RenderQuerySpecialtyType::kLines:
        throw std::runtime_error(
            "Line or poly types are currently not supported for simulating query "
            "execution.");
      case RenderQuerySpecialtyType::kNone:
#ifdef HAVE_CUDA
        std::tie(vbolayout, ubolayout) = runCuda(render_manager_, query_str);
#else   // !HAVE_CUDA
        std::tie(vbolayout, ubolayout) = runCPUQuery(render_manager_, query_str);
#endif  // HAVE_CUDA
        break;
      case RenderQuerySpecialtyType::kMesh2d:
        CHECK(false) << "Not implemented";
        break;
    }
    auto tmpresults = std::make_shared<ResultSet>(0, 0, nullptr);
    TargetEntries tmptargetentries;
    SQLSelectedTableContainer tableinfo;
    return {RenderQueryBufferLayouts(vbolayout, ubolayout),
            RenderQueryOutput(
                RenderQueryStructure(
                    std::move(tmptargetentries), std::move(tableinfo), insitu_flags),
                RenderQueryResult(std::move(tmpresults), 0, insitu_flags))};
  }

  std::vector<int32_t> getStringIds(const std::string& col_name,
                                    const std::vector<std::string>& col_vals,
                                    const QueryDataLayout* query_data_layout,
                                    const ResultSet* results,
                                    const bool warn = false) const final {
    CHECK(false);
    return {};
  }

  std::vector<std::string> getStringsFromIds(const std::string& col_name,
                                             const std::vector<int32_t>& ids,
                                             const QueryDataLayout* query_data_layout,
                                             const ResultSet* results) const final {
    CHECK(false);
    return {};
  }

 private:
  QueryRenderManager& render_manager_;
  RenderInfo render_info_;
};

int main(int argc, char* argv[]) {
  std::string outputprefix = "output_render";
  namespace po = boost::program_options;
  po::options_description desc("Options");
  desc.add_options()("help,h", "Print help messages");
  desc.add_options()("output-prefix,o",
                     po::value<std::string>(&outputprefix)->default_value(outputprefix),
                     "Output prefix of the resulting pngs.");
  desc.add_options()(
      "inputs,i", po::value<std::vector<std::string>>(), "Input vega files to render");

  logger::LogOptions log_options(argv[0]);
  log_options.max_files_ = 0;  // stderr only by default
  desc.add(log_options.get_options());

  po::positional_options_description p;
  p.add("inputs", -1);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }

  if (!vm.count("inputs")) {
    std::cout << "At least 1 vega input file is required." << std::endl << std::endl;
    std::cout << desc << std::endl;
    return 1;
  }

  auto inputVegas = vm["inputs"].as<std::vector<std::string>>();

  logger::init(log_options);

  // setup a custom log dir?
  // FLAGS_log_dir = "./LOGS";

#ifdef HAVE_CUDA
  const bool use_vulkan_driver = false;
  auto tmp_name = boost::filesystem::temp_directory_path();
  tmp_name /= boost::filesystem::unique_path("VegaRenderTest-%%%%");
  std::cerr << "CROOT - tmp data path to appease datamgr: " << tmp_name << std::endl;
  Data_Namespace::DataMgr dataMgr(
      tmp_name.string(),
      SystemParameters(),
      std::make_unique<CudaMgr_Namespace::CudaMgr>(
          use_vulkan_driver ? 1 : -1, 0),  // cannot use multi-gpus with vulkan (yet)
      true,
      -1);
  std::unique_ptr<QueryRenderManager> renderManager(new QueryRenderManager(
      nullptr, &dataMgr, 500000, 1500, 300000u, false, gfx::RasterSampleCount{4}, false));
#else
  std::unique_ptr<QueryRenderManager> renderManager(new QueryRenderManager(
      nullptr, nullptr, 500000, 1500, 300000u, false, gfx::RasterSampleCount{4}, false));
#endif

  const std::string userId = "1";
  const QueryRenderer::WidgetId widgetId = 1;
  auto session_info = std::make_shared<Catalog_Namespace::SessionInfo>(
      nullptr, Catalog_Namespace::UserMetadata(), ExecutorDeviceType::CPU, userId);

  for (size_t i = 0; i < inputVegas.size(); ++i) {
    const auto& vega = inputVegas[i];
    try {
      std::string configJSON;
      std::ifstream inFile(vega);
      inFile.seekg(0, std::ios::end);
      configJSON.reserve(inFile.tellg());
      inFile.seekg(0, std::ios::beg);
      configJSON.assign((std::istreambuf_iterator<char>(inFile)),
                        std::istreambuf_iterator<char>());
      inFile.close();

      auto const& render_session = renderManager->getOrCreateRenderSession(
          session_info, widgetId, std::move(configJSON));

      auto renderInfo = renderManager->runRenderRequest(
          render_session,
          std::make_unique<RenderQueryRunner>(render_session.getKey(), *renderManager));

      auto const& pixels = renderInfo.renderData;
      auto const image = QueryRenderer::QueryRenderManager::encodePNG(pixels, 5);

      std::ofstream pngFile(outputprefix + "_" + std::to_string(i) + ".png",
                            std::ios::binary);
      pngFile.write(image.c_str(), image.size());
      pngFile.close();
    } catch (std::exception& err) {
      std::cout << "Error trying to render vega from file: " << vega << ". "
                << err.what();
    }
  }

  return 0;
}
