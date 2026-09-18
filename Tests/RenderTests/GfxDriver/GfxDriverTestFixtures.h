/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * @file    GfxDriverTestFixtures.h
 * @author  Steve Blackmon <steve.blackmon@omnisci.com>
 * @description Test fixtures to simplify authoring of low level GfxDriver tests
 *
 * Usage:
 *  - Based on Google Test 1.10. Please read the docs:
 *    https://github.com/google/googletest/tree/release-1.10.0/googletest/docs
 *    especially advanced.md to get the most of these fixtures.
 *
 *  - Fixtures are templated on google test base class, driver type, and shader library
 *    content and make heavy use of alias templates, both for convenience and for
 *    compatibility with the gtest infrastructure (most of the test instantiation
 *    macros fail if they contain a template instantiation, use an alias instead.)
 *
 *  - Typed Tests are supported to allow authoring single tests that are run on
 *    multiple drivers. For now, we only have Vulkan, but the typed test mechanism
 *    remains for future use
 *    https://github.com/google/googletest/blob/release-1.10.0/googletest/docs/advanced.md#typed-tests
 *
 *    Example tests in GfxDriverTest.cpp:
 *      Driver: TYPED_TEST(TypedDriverTest, CanCreateDeviceContexts)
 *      DeviceContext: TYPED_TEST(TypedDeviceContextTest, GetDeviceProperties) including
 *      using if constexpr to constrain code to a specific driver
 *
 *  - Value-Parameterized Tests are supported. A gtest limitation is that these are
 *    mutually exclusive with Typed Tests so will need to be explicitly instantiated for
 *    both driver types
 *    https://github.com/google/googletest/blob/release-1.10.0/googletest/docs/advanced.md#how-to-write-value-parameterized-tests
 *
 *    Example test in GfxDriverTest.cpp
 *      VulkanMemoryCreateWithUsageBitTest - instantiates tests for allocating buffers
 *      with various usage bits from a single test declaration. (This would benefit
 *      from shared test resources - see TODO note below)
 *
 *      ShaderCompilerTest.cpp demonstrates a more complex usage of value-parameterized
 *      tests though it does not require these fixtures.
 *
 *  - ShaderManager's library is empty by default, but can be customized using policies.
 *    An example is GfxDriverTestShaderLibrary which is used by Gfx tests that require
 *    shaders These could be simple lambdas but using a class affords more options for
 *    customization. Example test in GfxDriverTest.cpp CanCreateGLShaderWithBuilders
 *      TODO: example using inline shader strings
 *
 * Death Tests:
 *   https://github.com/google/googletest/blob/release-1.10.0/googletest/docs/advanced.md#death-tests
 *
 *  - Make death tests explicit in the test name and keep them separate from non-death
 *    tests. There are special aliases for the fixtures that should be used as well to
 *    document that a test contains death assertions.
 *
 * TODO:
 *  - Move storage of the DriverInstance and DeviceContexts into policy classes.
 *    This will enable use of persistent driver and devices across multiple tests using
 *    SetUpTestSuite() and TearDownTestSuite() static functions
 *    https://github.com/google/googletest/blob/release-1.10.0/googletest/docs/advanced.md#sharing-resources-between-tests-in-the-same-test-suite
 */

#pragma once

#include <gtest/gtest.h>

#include <numeric>

#include "GfxDriver/DriverInstance.h"
#include "GfxDriver/Drivers/Vulkan/VulkanDriver.h"
#include "GfxDriver/GfxContext.h"
#include "GfxDriver/ShaderCompiler/Library.h"

using namespace gfx;
using DeviceContextVector = std::vector<DeviceContextUqPtr>;
using UUIDVector = std::vector<heavyai::UUID>;
using DeviceIdVector = std::vector<DeviceId>;

namespace GfxDriverTests {

#ifdef HAVE_CUDA
static constexpr GfxUsage kGfxUsage = GfxUsage::kCudaInterop;
#else
static constexpr GfxUsage kGfxUsage = GfxUsage::kSingleGpuPreferDiscreet;
#endif

/*
 * Shader library initializers
 */
class EmptyShaderLibrary {
 public:
  static void init(Library& library) {}
};

class GfxDriverTestShaderLibrary {
 public:
  static void init(Library& library) {
    library.addFromManifestFile("ShaderManifest.json",
                                std::string(RENDER_TESTS_PATH) + "GfxDriver/shaders/");
    library.addFromManifestFile("ShaderManifest.json",
                                std::string(GFX_DRIVER_PATH) + "Render/shaders/");
  }
};

/*
 * GfxContext factory functions for typed test fixtures
 */
template <typename DRIVER_TYPE>
// Typed tests only work with typenames, so we need to a conversion from the
// template typename to the enum type
struct DriverTypeEnum {
  static constexpr DriverType type_enum = DriverType::kVulkan;
};

template <typename DRIVER_TYPE, typename SHADER_LIBRARY = EmptyShaderLibrary>
struct GfxContextFactory {
  static std::unique_ptr<GfxContext> create() {
    auto library = std::make_unique<Library>();
    SHADER_LIBRARY::init(*library);
    auto driver_type = DriverTypeEnum<DRIVER_TYPE>::type_enum;

    // temporary env-var to control ray-tracing initialization
    // here, the default must be true
    // please forgive any and all double negatives
    auto* heavyai_allow_raytracing_init = getenv("HEAVYAI_ALLOW_RAYTRACING_INIT");
    if (heavyai_allow_raytracing_init) {
      std::cout << "**** DEBUG **** GfxContextFactory: HEAVYAI_ALLOW_RAYTRACING_INIT = "
                << heavyai_allow_raytracing_init << std::endl;
    }
    const bool allow_raytracing_init =
        heavyai_allow_raytracing_init ? (std::stoi(heavyai_allow_raytracing_init) == 1)
                                      : true;

    return std::make_unique<GfxContext>(driver_type,
                                        kGfxUsage,
                                        std::move(library),
                                        60000u,
                                        nullptr,
                                        allow_raytracing_init);
  }
};

using DriverTypes = ::testing::Types<VulkanDriver>;

/*
 * Driver test fixture
 * Can create DeviceContexts but doesn't create any on its own. Useful for testing context
 * creation API but is primarily a base for other tests
 */
template <typename GTEST_BASE,
          typename DRIVER_TYPE,
          typename SHADER_LIBRARY = EmptyShaderLibrary>
class DriverTest : public GTEST_BASE {
 protected:
  void SetUp() override {
    // instantiate the gfxcontext and get the driver
    gfx_context_ = GfxContextFactory<DRIVER_TYPE, SHADER_LIBRARY>::create();
    ASSERT_NE(nullptr, gfx_context_) << "Failed to create GfxContext";
    driver_ = &gfx_context_->getPrimaryDriver();
    ASSERT_NE(nullptr, driver_) << "Failed to create DriverInstance";

    // Get uuids for installed GPUs
    this->uuids_ = this->driver_->getUUIDs();
    ASSERT_NE(0UL, this->uuids_.size()) << "No UUIDs returned from getUUIDs(), no valid "
                                        << this->driver_->getName() << " devices?";

    // Generate mock cuda IDs starting at index 1 to ensure a value was set
    this->mock_cuda_ids_.resize(this->uuids_.size());
    std::iota(this->mock_cuda_ids_.begin(), this->mock_cuda_ids_.end(), 1);
  }

  void TearDown() override {
    ASSERT_NO_THROW(gfx_context_ = nullptr) << "GfxContext destructor threw an exception";
  }

 protected:
  // Utility functions
  [[nodiscard]] DeviceContextVector createDeviceContexts(
      const UUIDVector& uuids,
      const DeviceIdVector& device_ids) {
    DeviceContextVector device_contexts;
    CHECK_EQ(uuids.size(), device_ids.size());
    for (size_t i = 0; i < uuids.size(); ++i) {
      device_contexts.push_back(
          this->driver_->createDeviceContext(uuids[i], device_ids[i]));
    }
    return device_contexts;
  }

  void destroyDeviceContexts(DeviceContextVector& device_contexts) {
    for (auto& device : device_contexts) {
      bool did_leak_resources = false;
      if (device) {
        EXPECT_NO_THROW(did_leak_resources =
                            this->driver_->destroyDeviceContext(std::move(device)));
        EXPECT_FALSE(did_leak_resources) << "Failed to explicitly destroy all resources";
      }
    }
    device_contexts.clear();
    device_contexts.shrink_to_fit();
  }

  std::unique_ptr<GfxContext> gfx_context_;
  const DriverInstance* driver_ = nullptr;
  UUIDVector uuids_;
  DeviceIdVector mock_cuda_ids_;
};

template <typename DRIVER_TYPE>
using TypedDriverTest = DriverTest<::testing::Test, DRIVER_TYPE, EmptyShaderLibrary>;

// Instantiate test fixture using our type list. Any test based on this alias
// will automatically be instantiated for all known driver types.
TYPED_TEST_SUITE(TypedDriverTest, DriverTypes);

/*
 * DeviceContext test fixture
 * Initializes the driver and creates device contexts for each GPU in the system.
 * Setup and Teardown are performed for each test, so the DeviceContexts are pristine.
 *
 * Most tests should be based on this fixture, or one of the driver specific aliases
 * if testing a specific API
 */
template <typename GTEST_BASE,
          typename DRIVER_TYPE,
          typename SHADER_LIBRARY = EmptyShaderLibrary>
class DeviceContextTest : public DriverTest<GTEST_BASE, DRIVER_TYPE, SHADER_LIBRARY> {
 protected:
  using Base = DriverTest<GTEST_BASE, DRIVER_TYPE, SHADER_LIBRARY>;

  void SetUp() override {
    // Call base class SetUp to initialize the driver
    Base::SetUp();

    ASSERT_NO_THROW(this->device_contexts_ =
                        this->createDeviceContexts(this->uuids_, this->mock_cuda_ids_));
  }

  void TearDown() override {
    // Destroy device contexts before tearing down the fixture
    this->destroyDeviceContexts(this->device_contexts_);

    // Call base class to explicitly teardown the driver
    Base::TearDown();
  }

  void recreateDeviceContexts() {
    ASSERT_NO_THROW(this->destroyDeviceContexts(this->device_contexts_));
    ASSERT_NO_THROW(this->device_contexts_ =
                        this->createDeviceContexts(this->uuids_, this->mock_cuda_ids_));
  }

  DeviceContextVector device_contexts_;
};

/*
 * Typed DeviceContext tests
 */

// Instantiate test fixture using our type list. Any test based on this alias
// will automatically be instantiated for all driver types.
template <typename DRIVER_TYPE>
using TypedDeviceContextTest =
    DeviceContextTest<::testing::Test, DRIVER_TYPE, EmptyShaderLibrary>;
TYPED_TEST_SUITE(TypedDeviceContextTest, DriverTypes);

// Explicit driver type tests. Use for bulk testing of Vulkan specific features
using VulkanDeviceContextTest = DeviceContextTest<::testing::Test, VulkanDriver>;
using VulkanDeviceContextDeathTest = VulkanDeviceContextTest;

/*
 * Value-parameterized tests
 *
 * NOTE: the base class changes which prevents these from being used as typed tests.
 */
template <typename PARAM_TYPE,
          typename DRIVER_TYPE,
          typename SHADER_LIBRARY = EmptyShaderLibrary>
using DeviceContextTestWithParam =
    DeviceContextTest<::testing::TestWithParam<PARAM_TYPE>, DRIVER_TYPE, SHADER_LIBRARY>;

template <typename PARAM_TYPE, typename SHADER_LIBRARY = EmptyShaderLibrary>
using VulkanDeviceContextTestWithParam =
    DeviceContextTestWithParam<PARAM_TYPE, VulkanDriver, SHADER_LIBRARY>;

}  // namespace GfxDriverTests
