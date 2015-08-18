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
