#pragma once

#include "gpu/vulkan/context.h"
#include "gpu/vulkan/shader.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace gpu::vulkan {

class CommandBatch;

inline constexpr uint32_t kDescriptorSetIndex = 0;
inline constexpr VkDeviceSize kMinimumConstantPayloadBytes = kVertexRegisterBytes;
inline constexpr uint32_t kDynamicConstantBindingCount = kConstantBindingCount;

struct DescriptorSet {
    VkDescriptorSet handle = VK_NULL_HANDLE;
    uint64_t generation = 0;
    const void* owner = nullptr;
    VkDeviceSize constant_buffer_size = 0;
    VkDeviceSize constant_range = 0;

    VkDescriptorSet Handle() const { return handle; }
    bool Empty() const { return handle == VK_NULL_HANDLE; }
};

struct DescriptorResources {
    VkBuffer constant_buffer = VK_NULL_HANDLE;
    VkDeviceSize constant_buffer_size = 0;
    VkDeviceSize constant_range = 0;
    std::array<VkImageView, kTextureBindingCount> textures{};
    std::array<VkSampler, kSamplerBindingCount> samplers{};
    VkImageLayout texture_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
};

// Fixed set-0 ABI: b0..b7 are dynamic UBOs, t0..t15 map to 16..31, and
// s0..s15 map to 32..47. The gaps are intentional and consume no descriptors.
class DescriptorArena {
public:
    DescriptorArena() = default;
    DescriptorArena(const DescriptorArena&) = delete;
    DescriptorArena& operator=(const DescriptorArena&) = delete;
    ~DescriptorArena();

    bool Init(const Context& context, uint32_t sets_per_pool, std::string& error);
    void Reset();

    bool Allocate(const DescriptorResources& resources, DescriptorSet& set, std::string& error);
    bool Bind(VkCommandBuffer command, VkPipelineLayout pipeline_layout, const DescriptorSet& set,
              const std::array<uint32_t, kDynamicConstantBindingCount>& dynamic_offsets,
              std::string& error) const;

    // The owner calls MarkInFlight immediately after submitting a batch that uses
    // these descriptors. ResetAfterCompletion waits on that exact owner before
    // resetting pools, so stale set wrappers cannot be reused accidentally.
    bool MarkInFlight(const CommandBatch& batch, std::string& error);
    bool ResetAfterCompletion(CommandBatch& batch, uint64_t timeout_ns, std::string& error);

    VkDescriptorSetLayout Layout() const { return layout_; }
    VkDevice Device() const { return device_; }
    uint64_t Generation() const { return generation_; }
    uint32_t SetsPerPool() const { return sets_per_pool_; }
    size_t PoolCount() const { return pools_.size(); }
    bool IsCurrent(const DescriptorSet& set) const;
    bool Validate(const DescriptorSet& set,
                  const std::array<uint32_t, kDynamicConstantBindingCount>& dynamic_offsets,
                  std::string& error) const;

private:
    struct Pool {
        VkDescriptorPool handle = VK_NULL_HANDLE;
        uint32_t allocated = 0;
    };

    bool CreatePool(std::string& error);
    bool ValidateResources(const DescriptorResources& resources, std::string& error) const;
    bool ResetPools(std::string& error);

    VkDevice device_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout layout_ = VK_NULL_HANDLE;
    uint32_t sets_per_pool_ = 0;
    size_t pool_at_ = 0;
    uint64_t generation_ = 1;
    const CommandBatch* in_flight_batch_ = nullptr;
    VkDeviceSize alignment_ = 1;
    VkDeviceSize max_uniform_range_ = 0;
    PFN_vkCreateDescriptorSetLayout create_layout_ = nullptr;
    PFN_vkDestroyDescriptorSetLayout destroy_layout_ = nullptr;
    PFN_vkCreateDescriptorPool create_pool_ = nullptr;
    PFN_vkDestroyDescriptorPool destroy_pool_ = nullptr;
    PFN_vkResetDescriptorPool reset_pool_ = nullptr;
    PFN_vkAllocateDescriptorSets allocate_sets_ = nullptr;
    PFN_vkUpdateDescriptorSets update_sets_ = nullptr;
    PFN_vkCmdBindDescriptorSets bind_sets_ = nullptr;
    std::vector<Pool> pools_;
};

}  // namespace gpu::vulkan
