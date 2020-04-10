/*
 * Copyright 2019 OmniSci, Inc.
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

/**
 * @file    file_delete.h
 * @author  michael@omnisci.com>
 * @brief   shared utility for the db server and string dictionary server to remove old
 * files
 *
 */

#pragma once

// this is to clean up the deleted files
void file_delete(std::atomic<bool>& program_is_running,
                 const unsigned int wait_interval_seconds,
                 const std::string base_path);
