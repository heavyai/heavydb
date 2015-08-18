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
  int socketDepth = hwloc_get_type_depth(topology, HWLOC_OBJ_SOCKET);
  hwloc_topology_init(&topology);
  hwloc_topology_load(topology);
  if (socketDepth == HWLOC_TYPE_DEPTH_UNKNOWN) {
  } else {
    hwloc_get_nbobjs_by_depth(topology, socketDepth);
  }
}

void SystemProfiler::profileSystem(const std::string& dataDir) {
  rootNode_ = new SystemNode;
  rootNode_->nodeType = STORAGE_NODE;
  boost::filesystem::path path(dataDir);
  boost::filesystem::space_info spaceInfo = boost::filesystem::space(path);
  rootNode_->memCapacity = spaceInfo.capacity;
  rootNode_->memFree = spaceInfo.available;
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
  printNode(startNode);
  for (vector<SystemNode*>::const_iterator childNodeIt = startNode->childNodes.begin();
       childNodeIt != startNode->childNodes.end();
       ++childNodeIt) {
    printTree(*childNodeIt);
  }
}

int main() {
  SystemProfiler("data");
}
