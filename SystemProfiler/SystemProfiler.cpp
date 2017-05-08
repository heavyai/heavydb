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

#include "SystemProfiler.h"

#include <vector>
#include <iostream>
#include <boost/filesystem.hpp>
#include <hwloc.h>

using namespace std;

SystemProfiler::SystemProfiler(const std::string& dataDir) {
  profileSystem(dataDir);
}

SystemProfiler::~SystemProfiler() {
  deleteNodes(rootNode_);
}

void SystemProfiler::deleteNodes(SystemNode* startNode) {
  // recursively descends the tree, deleting on the way up
  for (vector<SystemNode*>::iterator childNodeIt = startNode->childNodes.begin();
       childNodeIt != startNode->childNodes.end();
       ++childNodeIt) {
    deleteNodes(*childNodeIt);
  }
  delete startNode;
}

void SystemProfiler::addNumaNodes(SystemNode* parentNode) {
  hwloc_topology_t topology;
  hwloc_topology_init(&topology);
  hwloc_topology_load(topology);
  int socketDepth = hwloc_get_type_depth(topology, HWLOC_OBJ_NODE);
  hwloc_topology_init(&topology);
  hwloc_topology_load(topology);
  if (socketDepth == HWLOC_TYPE_DEPTH_UNKNOWN) {
    cout << "Depth unknown" << endl;
  } else {
    int numSockets = hwloc_get_nbobjs_by_depth(topology, socketDepth);
    cout << "Num sockets: " << numSockets << endl;
    for (int socketNum = 0; socketNum != numSockets; ++socketNum) {
      SystemNode* socketNode = new SystemNode;
      hwloc_obj_t socket = hwloc_get_obj_by_depth(topology, socketDepth, socketNum);
      socketNode->nodeType = CPU_NODE;
      socketNode->memCapacity = socket->memory.total_memory;
      socketNode->memFree = socket->memory.local_memory;
      hwloc_obj_t lastChild = socket->last_child;
      int coreCount = 0;
      while (socket->children[coreCount] != lastChild) {
        coreCount++;
      }
      coreCount++;
      cout << "Core count: " << coreCount << endl;

      parentNode->childNodes.push_back(socketNode);
      nodeLevelMap_[CPU_NODE].push_back(socketNode);
    }
  }
}

void SystemProfiler::addStorage(const std::string& dataDir) {
  rootNode_ = new SystemNode;
  rootNode_->nodeType = STORAGE_NODE;
  boost::filesystem::space_info spaceInfo = boost::filesystem::space(dataDir);
  rootNode_->memCapacity = spaceInfo.capacity;
  rootNode_->memFree = spaceInfo.available;

  nodeLevelMap_[STORAGE_NODE].push_back(rootNode_);
}

void SystemProfiler::profileSystem(const std::string& dataDir) {
  addStorage(dataDir);
  addNumaNodes(rootNode_);
}

void SystemProfiler::printNode(const SystemNode* node) {
  cout << "NodeType: ";
  switch (node->nodeType) {
    case STORAGE_NODE:
      cout << " Storage" << endl;
      break;
    case CPU_NODE:
      cout << " Cpu" << endl;
      break;
    case GPU_NODE:
      cout << " GPU" << endl;
      break;
  }
  cout << "Mem capacity: " << node->memCapacity << endl;
  cout << "Mem free: " << node->memFree << endl;
}

void SystemProfiler::printTree(const SystemNode* startNode) {
  if (startNode == 0) {
    printTree(rootNode_);
  } else {
    printNode(startNode);
    for (vector<SystemNode*>::const_iterator childNodeIt = startNode->childNodes.begin();
         childNodeIt != startNode->childNodes.end();
         ++childNodeIt) {
      printTree(*childNodeIt);
    }
  }
}

int main() {
  SystemProfiler systemProfiler("data");
  systemProfiler.printTree();
}
