// @author Todd Mostak <todd@map-d.com>

#include "SystemProfiler.h"

#include <vector>
#include <boost/filesystem.hpp>


SystemProfiler::SystemProfiler(const std::string &dataDir) {
    profileSystem(dataDir);
}

SystemProfiler::~SystemProfiler() {
    deleteNodes();
}


void SystemProfiler::deleteNodes(SystemNode *startNode) {
    //recursively descends the tree, deleting on the way up
    for (auto childNodeIt = startNode -> childNodes.begin(); childNodeIt != startNode -> childNodes.end(); ++childNodeIt) {
        deleteNodes(*childNodeIt);
    }
    delete startNode;
}

void SystemProfiler::profileSystem(const std::string &dataDir) {
    rootNode_ = new SystemNode;
    rootNode_ -> nodeType = STORAGE_NODE; 
    boost::filesystem::spaceinfo spaceInfo = boost::filesystem::space(dataDir);
    rootNode_ -> memCapacity = spaceInfo.capacity;
    rootNode_ -> memFree = spaceInfo.available;







}















