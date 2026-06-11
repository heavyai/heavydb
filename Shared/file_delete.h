/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file    file_delete.h
 * @brief   shared utility for the db server and string dictionary server to remove old
 * files
 *
 */

#pragma once

// this is to clean up the deleted files
void file_delete(std::atomic<bool>& program_is_running,
                 const unsigned int wait_interval_seconds,
                 const std::string base_path);
