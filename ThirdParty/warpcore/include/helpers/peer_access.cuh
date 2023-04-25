#ifndef HELPERS_PEER_ACCESS_CUH
#define HELPERS_PEER_ACCESS_CUH

#ifdef __NVCC__

    #include <cassert>
    #include <iostream>
    #include <vector>

    #include "cuda_helpers.cuh"

    namespace helpers {

    enum class PeerAccessDebugMode {Enabled, Disabled};

    template<PeerAccessDebugMode dbg>
    struct PeerAccessBase{
        static constexpr PeerAccessDebugMode debugmode = dbg;

        bool resetOnDestruction;
        int numGpus;
        std::vector<int> deviceIds;
        std::vector<int> accessMatrix;
        std::vector<int> oldEnabledPeerAccesses;

        PeerAccessBase(){
            int numDevices = 0;
            cudaGetDeviceCount(&numDevices);

            std::vector<int> ids(numDevices);
            for(int i = 0; i < numDevices; i++){
                ids[i] = i;
            }

            init(std::move(ids), true);
        }

        PeerAccessBase(std::vector<int> deviceIds_, bool resetOnDestruction_){
            init(std::move(deviceIds_), resetOnDestruction_);
        }

        void init(std::vector<int> deviceIds_, bool resetOnDestruction_){
            deviceIds = std::move(deviceIds_);
            resetOnDestruction = resetOnDestruction_;
            cudaGetDeviceCount(&numGpus);

            accessMatrix.resize(numGpus * numGpus);
            const int numIds = deviceIds.size();
            for(int i = 0; i < numIds; i++){
                for(int k = 0; k < numIds; k++){
                    //device i can access device k?
                    const int dev1 = deviceIds[i];
                    const int dev2 = deviceIds[k];
                    cudaDeviceCanAccessPeer(&accessMatrix[dev1 * numGpus + dev2], dev1, dev2); CUERR;
                    if(debugmode == PeerAccessDebugMode::Enabled){
                        std::cerr << "Peer access possible for " << dev1 << " -> " << dev2 << "\n";
                    }
                }
            }

            if(resetOnDestruction){
                //save current enabled peer accesses
                oldEnabledPeerAccesses = getEnabledPeerAccesses();
            }
        }

        ~PeerAccessBase(){
            if(resetOnDestruction && int(oldEnabledPeerAccesses.size()) == numGpus * numGpus){
                setEnabledPeerAccesses(oldEnabledPeerAccesses);
            }
        }

        PeerAccessBase(const PeerAccessBase&) = default;
        PeerAccessBase(PeerAccessBase&&) = default;
        PeerAccessBase& operator=(const PeerAccessBase&) = default;
        PeerAccessBase& operator=(PeerAccessBase&&) = default;

        bool canAccessPeer(int device, int peerDevice) const{
            assert(device < numGpus);
            assert(peerDevice < numGpus);

            return accessMatrix[device * numGpus + peerDevice] == 1;
        }

        void enablePeerAccess(int device, int peerDevice) const{
            if(!canAccessPeer(device, peerDevice)){
                if(debugmode == PeerAccessDebugMode::Enabled){
                    std::cerr << "Peer access from " << device << " to " << peerDevice << " is not available and cannot be enabled.\n";
                }
                return;
            }

            int oldId; cudaGetDevice(&oldId); CUERR;
            cudaSetDevice(device); CUERR;
            cudaError_t status = cudaDeviceEnablePeerAccess(peerDevice, 0);
            if(status != cudaSuccess){
                if(status == cudaErrorPeerAccessAlreadyEnabled){
                    if(debugmode == PeerAccessDebugMode::Enabled){
                        std::cerr << "Peer access from " << device << " to " << peerDevice << " has already been enabled. This is not a program error\n";
                    }
                    cudaGetLastError(); //reset error state;
                }else{
                    CUERR;
                }
            }
            cudaSetDevice(oldId); CUERR;
        }

        void disablePeerAccess(int device, int peerDevice) const{
            if(!canAccessPeer(device, peerDevice)){
                if(debugmode == PeerAccessDebugMode::Enabled){
                    std::cerr << "Peer access from " << device << " to " << peerDevice << " is not available and cannot be disabled.\n";
                }
                return;
            }

            int oldId; cudaGetDevice(&oldId); CUERR;
            cudaSetDevice(device); CUERR;
            cudaError_t status = cudaDeviceDisablePeerAccess(peerDevice);
            if(status != cudaSuccess){
                if(status == cudaErrorPeerAccessNotEnabled){
                    if(debugmode == PeerAccessDebugMode::Enabled){
                        std::cerr << "Peer access from " << device << " to " << peerDevice << " has not yet been enabled. This is not a program error\n";
                    }
                    cudaGetLastError(); //reset error state;
                }else{
                    CUERR;
                }
            }
            cudaSetDevice(oldId); CUERR;
        }

        void enableAllPeerAccesses(){
            for(int i = 0; i < numGpus; i++){
                for(int k = 0; k < numGpus; k++){
                    if(canAccessPeer(i, k)){
                        enablePeerAccess(i, k);
                    }
                }
            }
        }

        void disableAllPeerAccesses(){
            for(int i = 0; i < numGpus; i++){
                for(int k = 0; k < numGpus; k++){
                    if(canAccessPeer(i, k)){
                        disablePeerAccess(i, k);
                    }
                }
            }
        }

        std::vector<int> getEnabledPeerAccesses() const{
            int numGpus = 0;
            cudaGetDeviceCount(&numGpus); CUERR;

            std::vector<int> result(numGpus * numGpus, 0);

            if(numGpus > 0){
                int oldId; cudaGetDevice(&oldId); CUERR;

                for(int i = 0; i < numGpus; i++){
                    cudaSetDevice(i); CUERR;
                    for(int k = 0; k < numGpus; k++){
                        if(canAccessPeer(i,k)){
                            cudaError_t status = cudaDeviceDisablePeerAccess(k);
                            if(status == cudaSuccess){
                                //if device i can disable access to device k, it must have been enabled
                                result[i * numGpus + k] = 1;
                                //enable again
                                cudaDeviceEnablePeerAccess(k, 0); CUERR;
                            }else{
                                if(status != cudaErrorPeerAccessNotEnabled){
                                    CUERR; //error
                                }
                                cudaGetLastError(); //reset error state;
                            }
                        }
                    }
                }

                cudaSetDevice(oldId);
            }

            return result;
        }

        std::vector<int> getDisabledPeerAccesses() const{
            std::vector<int> result = getEnabledPeerAccesses();
            for(auto& i : result){
                i = (i == 0) ? 1 : 0; // 0->1, 1->0
            }
            return result;
        }

        void setEnabledPeerAccesses(const std::vector<int>& vec){
            for(int i = 0; i < numGpus; i++){
                for(int k = 0; k < numGpus; k++){
                    if(canAccessPeer(i,k)){
                        int flag = vec[i * numGpus + k];
                        if(flag == 1){
                            enablePeerAccess(i,k);
                        }else{
                            disablePeerAccess(i,k);
                        }
                    }
                }
            }
        }
    };

    using PeerAccess = PeerAccessBase<PeerAccessDebugMode::Disabled>;
    using PeerAccessDebug = PeerAccessBase<PeerAccessDebugMode::Enabled>;

    } // namespace helpers

#endif

#endif /* HELPERS_PEER_ACCESS_CUH */

