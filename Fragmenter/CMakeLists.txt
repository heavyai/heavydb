# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2015-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

add_library(Fragmenter InsertOrderFragmenter.cpp SortedOrderFragmenter.cpp UpdelStorage.cpp TargetValueConvertersFactories.cpp InsertDataLoader.cpp RasterFragmenter.cpp)
add_dependencies(Fragmenter Calcite)
target_link_libraries(Fragmenter ${Boost_THREAD_LIBRARY})
