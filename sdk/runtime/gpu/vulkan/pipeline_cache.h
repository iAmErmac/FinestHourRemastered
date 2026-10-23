#pragma once

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <cstdint>
#include <filesystem>

namespace gpu::vulkan {

class Context;

struct PipelineCacheStats {
    bool enabled = false;
    bool loaded_from_disk = false;
    bool stored_to_disk = false;
    uint64_t loaded_bytes = 0;
    uint64_t stored_bytes = 0;
};

class PipelineCache {
public:
    PipelineCache() = default;
    PipelineCache(const PipelineCache&) = delete;
    PipelineCache& operator=(const PipelineCache&) = delete;
    ~PipelineCache();

    void Init(const Context& context, const std::filesystem::path& directory);
    bool Persist();
    void Reset(bool persist = true);

    VkPipelineCache Handle() const { return cache_; }
    const PipelineCacheStats& Stats() const { return stats_; }

private:
    const Context* context_ = nullptr;
    VkDevice device_ = VK_NULL_HANDLE;
    VkPipelineCache cache_ = VK_NULL_HANDLE;
    PFN_vkDestroyPipelineCache destroy_ = nullptr;
    PFN_vkGetPipelineCacheData get_data_ = nullptr;
    std::filesystem::path path_;
    PipelineCacheStats stats_{};
};

}  // namespace gpu::vulkan
