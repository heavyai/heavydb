/*
    Copyright 2021 OmniSci, Inc.
    Copyright (c) 2022 Intel Corporation
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#pragma once

enum class ExtModuleKinds {
  template_module,     // RuntimeFunctions.bc
  udf_cpu_module,      // Load-time UDFs for CPU execution
  udf_gpu_module,      // Load-time UDFs for GPU execution
  rt_udf_cpu_module,   // Run-time UDF/UDTFs for CPU execution
  rt_udf_gpu_module,   // Run-time UDF/UDTFs for GPU execution
  rt_libdevice_module  // math library functions for GPU execution
};
