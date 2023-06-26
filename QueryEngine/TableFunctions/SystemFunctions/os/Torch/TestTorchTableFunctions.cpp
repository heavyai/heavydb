/*
 * Copyright 2023 HEAVY.AI, Inc.
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

#include <cstdio>  // use fprintf because cout had weird concurrency issues
#include <cstdlib>
#include <ctime>

#include "TestTorchTableFunctions.h"

#undef LOG
#undef CHECK
#undef LOG_IF
#undef VLOG
#undef CHECK_OP
#undef CHECK_EQ
#undef CHECK_LT
#undef CHECK_LE
#undef CHECK_GT
#undef CHECK_GE
#undef CHECK_NE
#undef GLOBAL
#include "torch/script.h"
#include "torch/torch.h"

#ifndef __CUDACC__

torch::Device _test_torch_tfs_device = torch::kCPU;

EXTENSION_NOINLINE
int32_t tf_test_runtime_torch(TableFunctionManager& mgr,
                              Column<int64_t>& input,
                              Column<int64_t>& output) {
  return 0;
}

template <typename T>
TEMPLATE_NOINLINE int32_t
tf_test_runtime_torch_template__template(TableFunctionManager& mgr,
                                         const Column<T>& input,
                                         Column<T>& output) {
  return 0;
}

template TEMPLATE_NOINLINE int32_t
tf_test_runtime_torch_template__template(TableFunctionManager& mgr,
                                         const Column<int64_t>& input,
                                         Column<int64_t>& output);
template TEMPLATE_NOINLINE int32_t
tf_test_runtime_torch_template__template(TableFunctionManager& mgr,
                                         const Column<double>& input,
                                         Column<double>& output);

/* Generates a column of random values with @num_elements rows, using PyTorch's default
 * randn generator, which samples a normal distribution with mean 0 and variance 1. Is
 * used to generate data to be fed to the model implemented in tf_test_torch_regression.
 */
EXTENSION_NOINLINE int32_t tf_test_torch_generate_random_column(TableFunctionManager& mgr,
                                                                int32_t num_elements,
                                                                Column<double>& output) {
  mgr.set_output_row_size(num_elements);
  torch::Tensor random = torch::randn({num_elements}, at::dtype(at::kDouble));
  random = random.unsqueeze(1);
  double* data_ptr = (double*)random.data_ptr();

  for (int32_t i = 0; i < num_elements; ++i) {
    output[i] = *data_ptr++;
  }

  return num_elements;
}

torch::Tensor make_features_from_columns(const ColumnList<double>& cols,
                                         int32_t batch_size) {
  int32_t poly_degree = cols.numCols();
  torch::Tensor output = torch::empty({batch_size, poly_degree}, {torch::kCPU});

  // build a tensor of (batch_size, poly_degree) dimensions, where each row is sampled
  // randomly from the input columns formated as (x, x^2 ..., x^poly_degree)
  for (int i = 0; i < batch_size; i++) {
    int32_t idx = rand() % cols.size();
    for (int j = 0; j < poly_degree; j++) {
      output[i][j] = cols[j][idx];
    }
  }

  return output.to(_test_torch_tfs_device);
}

// Approximated function.
torch::Tensor f(torch::Tensor x, torch::Tensor W_target, torch::Tensor b_target) {
  return x.mm(W_target) + b_target.item();
}

// Creates a string description of a polynomial.
std::string poly_desc(torch::Tensor W, torch::Tensor b) {
  auto size = W.size(0);
  std::ostringstream stream;

  if (W.scalar_type() != c10::ScalarType::Float ||
      b.scalar_type() != c10::ScalarType::Float) {
    throw std::runtime_error(
        "Attempted to print polynomial with non-float coefficients!");
  }

  stream << "y = ";
  for (int64_t i = 0; i < size; ++i)
    stream << W[i].item<float>() << " x^" << size - i << " ";
  stream << "+ " << b[0].item<float>();
  return stream.str();
}

// Builds a batch i.e. (x, f(x)) pair.
std::pair<torch::Tensor, torch::Tensor> get_batch(const ColumnList<double>& cols,
                                                  torch::Tensor W_target,
                                                  torch::Tensor b_target,
                                                  int32_t batch_size) {
  auto x = make_features_from_columns(cols, batch_size);
  auto y = f(x, W_target, b_target);
  return std::make_pair(x, y);
}

/* This code is very heavily based on (in large part copy-pasted) from PyTorch's official
 * C++ API examples:
 * https://github.com/pytorch/examples/tree/main/cpp/regression. It trains
 * a single-layer Neural Network to fit a @poly_degree degree polynomial, using
 * @batch_size, and optionally using CUDA-powered libtorch, if available.
 * It optionally saves the model as a torchscript file with name @model_filename.
 * The code has been modified to generate feature data through LibTorch, store it in a
 * heavydb table, then pull data from that table to feed the model. It is very simplistic
 * and naive, particularly in how data is sampled from the generated data, but as a
 * proof-of-concept/example of how LibTorch can be used from within heavydb, it works.*/
EXTENSION_NOINLINE int32_t
tf_test_torch_regression(TableFunctionManager& mgr,
                         const ColumnList<double>& features,
                         int32_t batch_size,
                         bool use_gpu,
                         bool save_model,
                         const TextEncodingNone& model_filename,
                         Column<double>& output) {
  int32_t poly_degree = features.numCols();
  // we output target and trained coefficients + bias
  int32_t output_size = (poly_degree + 1) * 2;
  mgr.set_output_row_size(output_size);
  std::srand(std::time(nullptr));  // not ideal RNG, but fine for test purpooses
#ifdef HAVE_CUDA_TORCH
  if (torch::cuda::is_available() && use_gpu) {
    _test_torch_tfs_device = torch::kCUDA;
  }
#endif

  auto W_target = torch::randn({poly_degree, 1}, at::device(_test_torch_tfs_device)) * 5;
  auto b_target = torch::randn({1}, at::device(_test_torch_tfs_device)) * 5;

  // Define the model and optimizer
  auto fc = torch::nn::Linear(W_target.size(0), 1);
  fc->to(_test_torch_tfs_device);
  torch::optim::SGD optim(fc->parameters(), .1);

  float loss = 0;
  int64_t batch_idx = 0;

  while (++batch_idx) {
    // Get data
    torch::Tensor batch_x, batch_y;
    std::tie(batch_x, batch_y) = get_batch(features, W_target, b_target, batch_size);

    // Reset gradients
    optim.zero_grad();

    // Forward pass
    auto output = torch::smooth_l1_loss(fc(batch_x), batch_y);
    loss = output.item<float>();

    // Backward pass
    output.backward();

    // Apply gradients
    optim.step();

    // Stop criterion
    if (loss < 1e-3f)
      break;
  }

  if (save_model) {
    torch::save(fc, model_filename.getString());
  }

  // output column with target + trained coefficients ordered by degree, then bias
  torch::Tensor output_coefficients = fc->weight.view({-1}).cpu();
  torch::Tensor goal_coefficients = W_target.view({-1}).cpu();
  int32_t out_column_idx, input_idx;
  for (out_column_idx = 0, input_idx = 0; input_idx < output_coefficients.size(0);
       ++input_idx) {
    output[out_column_idx++] = output_coefficients[input_idx].item<float>();
    output[out_column_idx++] = goal_coefficients[input_idx].item<float>();
  }
  output[out_column_idx++] = fc->bias[0].item<float>();
  output[out_column_idx] = b_target[0].item<float>();

  std::fprintf(stdout, "Loss: %lf after %ld batches\n", loss, batch_idx);
  std::fprintf(stdout,
               "==> Learned function:\t%s\n",
               poly_desc(output_coefficients, fc->bias).c_str());
  std::fprintf(stdout,
               "==> Actual function:\t%s\n",
               poly_desc(W_target.view({-1}).cpu(), b_target).c_str());

  return output_size;
}

EXTENSION_NOINLINE int32_t
tf_test_torch_load_model(TableFunctionManager& mgr,
                         const TextEncodingNone& model_filename,
                         Column<bool>& output) {
  mgr.set_output_row_size(1);
  torch::jit::script::Module module;
  try {
    module = torch::jit::load(model_filename.getString());
  } catch (const std::exception& e) {
    return mgr.ERROR_MESSAGE("Error loading torchscript model: " + e.what());
  }

  output[0] = true;
  return 1;
}

#endif  // #ifndef __CUDACC__
