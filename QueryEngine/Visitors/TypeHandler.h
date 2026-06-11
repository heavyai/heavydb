/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file    TypeHandler.h
 * @brief   Sortable utility struct for visitor patterns.
 */

#pragma once

#include <typeindex>

template <typename T, typename U>
struct TypeHandler {
  std::type_index type_index;
  void (T::*handler)(U const*);
};

template <typename T, typename U>
bool operator<(TypeHandler<T, U> const& lhs, TypeHandler<T, U> const& rhs) {
  return lhs.type_index < rhs.type_index;
}

template <typename T, typename U>
bool operator<(TypeHandler<T, U> const& lhs, std::type_index const& rhs) {
  return lhs.type_index < rhs;
}

template <typename T, typename U>
bool operator<(std::type_index const& lhs, TypeHandler<T, U> const& rhs) {
  return lhs < rhs.type_index;
}
