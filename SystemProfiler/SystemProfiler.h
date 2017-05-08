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

// @author Todd Mostak <todd@map-d.com>

#ifndef SYSTEM_PROFILER_H
#define SYSTEM_PROFILER_H

#include <vector>
#include <map>
#include <string>

enum NodeType { STORAGE_NODE, CPU_NODE, GPU_NODE };

struct SystemNode {
  NodeType nodeType;
  int id;  // will correspond to cudaSetGpu(id) for GPUs
  size_t memCapacity;
  size_t memFree;
  int numCores;
  float coreSpeed;
  std::vector<SystemNode*> childNodes;
};

class SystemProfiler {
 public:
  SystemProfiler(const std::string& dataDir);
  ~SystemProfiler();
  void printTree(const SystemNode* startNode = 0);

 private:
  void profileSystem(const std::string& dataDir);
  void deleteNodes(SystemNode* startNode);
  void addStorage(const std::string& dataDir);
  void addNumaNodes(SystemNode* parentNode);
  void printNode(const SystemNode* node);
  SystemNode* rootNode_;
  std::map<NodeType, std::vector<SystemNode*>> nodeLevelMap_;
};

#endif  // SYSTEM_PROFILER_H
