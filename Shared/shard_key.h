/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef SHARED_SHARDKEY_H_
#define SHARED_SHARDKEY_H_

#define SHARD_FOR_KEY(key, num_shards) ((key % num_shards + num_shards) % num_shards)

#endif  // SHARED_SHARDKEY_H_
