/*
 * Copyright 2017 MapD Technologies, Inc.
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

#ifndef NOCUDA_H
#define NOCUDA_H

typedef int CUdevice;
typedef int CUresult;
typedef int CUcontext;
typedef void* CUmodule;
typedef void* CUfunction;
typedef int CUjit_option;
typedef int CUlinkState;
typedef unsigned long long CUdeviceptr;

#endif  // NOCUDA_H
