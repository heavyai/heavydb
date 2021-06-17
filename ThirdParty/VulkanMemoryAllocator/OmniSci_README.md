# Vulkan Memory Allocator

Easy to integrate Vulkan memory allocation library.

Upstream: https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator

> Last Edited: Matt Torok, Feb 24th 2021

## About this fork

Fork repo (public): https://github.com/omnisci/VulkanMemoryAllocator
 
This is an OmniSci fork of Vulkan Memory Allocator, branching from the official repo at hash `7eee5e3d262637400fe3b133f19025f980e20cad`.
 
The main change is to add support for exportable memory. This is surfaced by adding the `VMA_POOL_CREATE_EXPORTABLE_BIT` flag to the `VmaPoolCreateFlagBits` enum used when creating memory pools. When this flag is set during pool creation, the VkDeviceMemory block allocated for that pool will have the `VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO` struct passed during Vulkan allocation calls made by VMA. It is still the caller's responsibility to choose the correct memory type and other options to support memory export; all the flag does is cause the Vulkan exportable struct to be added to the Vulkan allocation options.
 
Due to the fact that VMA uses a slab allocator by default (a desirable property for us), the exportable option only works at the pool level. It makes all allocations in that pool exportable. You must then pass this pool to VMA allocation functions to allocate exportable memory. You can not make individual allocation exportable or un-exportable; only the pool.