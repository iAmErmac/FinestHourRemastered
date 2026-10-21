#include "gpu/vulkan/descriptors.h"

#include "gpu/vulkan/commands.h"
#include "core/log.h"

#include <algorithm>
#include <array>

namespace gpu::vulkan {
namespace {

constexpr VkShaderStageFlags kGraphicsStages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT |
                                               VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT |
                                               VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
constexpr size_t kMaxPoolCount = 64;

bool Result(VkResult result, const char* operation, std::string& error) {
    if (result == VK_SUCCESS) return true;
    error = std::string(operation) + " failed: VkResult " + std::to_string(static_cast<int>(result));
    return false;
}

}  // namespace

DescriptorArena::~DescriptorArena() { Reset(); }

bool DescriptorArena::Init(const Context& context, uint32_t sets_per_pool, std::string& error) {
    if (in_flight_batch_) {
        error = "Vulkan descriptor arena cannot be reinitialized while its owner batch is pending";
        return false;
    }
    Reset();
    error.clear();
    if (!context.Initialized() || sets_per_pool == 0 ||
        sets_per_pool > UINT32_MAX / kDynamicConstantBindingCount ||
        sets_per_pool > UINT32_MAX / kTextureBindingCount ||
        sets_per_pool > UINT32_MAX / kSamplerBindingCount) {
        error = "Vulkan descriptor arena requires an initialized context and nonzero pool capacity";
        return false;
    }
    device_ = context.Device();
    sets_per_pool_ = sets_per_pool;
    alignment_ = std::max<VkDeviceSize>(1, context.Properties().limits.minUniformBufferOffsetAlignment);
    max_uniform_range_ = context.Properties().limits.maxUniformBufferRange;
    create_layout_ = reinterpret_cast<PFN_vkCreateDescriptorSetLayout>(context.DeviceProc("vkCreateDescriptorSetLayout"));
    destroy_layout_ = reinterpret_cast<PFN_vkDestroyDescriptorSetLayout>(context.DeviceProc("vkDestroyDescriptorSetLayout"));
    create_pool_ = reinterpret_cast<PFN_vkCreateDescriptorPool>(context.DeviceProc("vkCreateDescriptorPool"));
    destroy_pool_ = reinterpret_cast<PFN_vkDestroyDescriptorPool>(context.DeviceProc("vkDestroyDescriptorPool"));
    reset_pool_ = reinterpret_cast<PFN_vkResetDescriptorPool>(context.DeviceProc("vkResetDescriptorPool"));
    allocate_sets_ = reinterpret_cast<PFN_vkAllocateDescriptorSets>(context.DeviceProc("vkAllocateDescriptorSets"));
    update_sets_ = reinterpret_cast<PFN_vkUpdateDescriptorSets>(context.DeviceProc("vkUpdateDescriptorSets"));
    bind_sets_ = reinterpret_cast<PFN_vkCmdBindDescriptorSets>(context.DeviceProc("vkCmdBindDescriptorSets"));
    if (!create_layout_ || !destroy_layout_ || !create_pool_ || !destroy_pool_ || !reset_pool_ || !allocate_sets_ ||
        !update_sets_ || !bind_sets_) {
        error = "Vulkan descriptor entry points are unavailable";
        Reset();
        return false;
    }

    std::array<VkDescriptorSetLayoutBinding, kConstantBindingCount + kTextureBindingCount + kSamplerBindingCount> bindings{};
    uint32_t at = 0;
    for (uint32_t i = 0; i < kDynamicConstantBindingCount; ++i, ++at) {
        bindings[at] = {i, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, kGraphicsStages, nullptr};
    }
    for (uint32_t i = 0; i < kTextureBindingCount; ++i, ++at) {
        bindings[at] = {kTextureBindingBase + i, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, kGraphicsStages, nullptr};
    }
    for (uint32_t i = 0; i < kSamplerBindingCount; ++i, ++at) {
        bindings[at] = {kSamplerBindingBase + i, VK_DESCRIPTOR_TYPE_SAMPLER, 1, kGraphicsStages, nullptr};
    }
    VkDescriptorSetLayoutCreateInfo create{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    create.bindingCount = static_cast<uint32_t>(bindings.size());
    create.pBindings = bindings.data();
    if (!Result(create_layout_(device_, &create, nullptr, &layout_), "vkCreateDescriptorSetLayout", error)) {
        Reset();
        return false;
    }
    return true;
}

void DescriptorArena::Reset() {
    // The public completion-aware reset is the required path while a submitted
    // batch owns descriptors. Refuse unsafe destruction; the owner can still
    // wait and call ResetAfterCompletion before context teardown.
    if (in_flight_batch_) {
        LOGE("vulkan: descriptor arena reset requested before its owner batch completed");
        return;
    }
    if (device_ != VK_NULL_HANDLE) {
        for (const Pool& pool : pools_)
            if (pool.handle && destroy_pool_) destroy_pool_(device_, pool.handle, nullptr);
        if (layout_ && destroy_layout_) destroy_layout_(device_, layout_, nullptr);
    }
    pools_.clear();
    device_ = VK_NULL_HANDLE;
    layout_ = VK_NULL_HANDLE;
    sets_per_pool_ = 0;
    pool_at_ = 0;
    alignment_ = 1;
    max_uniform_range_ = 0;
    in_flight_batch_ = nullptr;
    create_layout_ = nullptr;
    destroy_layout_ = nullptr;
    create_pool_ = nullptr;
    destroy_pool_ = nullptr;
    reset_pool_ = nullptr;
    allocate_sets_ = nullptr;
    update_sets_ = nullptr;
    bind_sets_ = nullptr;
    if (++generation_ == 0) generation_ = 1;
}

bool DescriptorArena::CreatePool(std::string& error) {
    if (!create_pool_ || !layout_ || !sets_per_pool_) {
        error = "Vulkan descriptor arena is not initialized";
        return false;
    }
    if (pools_.size() >= kMaxPoolCount) {
        error = "Vulkan descriptor arena reached its bounded pool limit";
        return false;
    }
    std::array<VkDescriptorPoolSize, 3> sizes = {
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, sets_per_pool_ * kDynamicConstantBindingCount},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, sets_per_pool_ * kTextureBindingCount},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLER, sets_per_pool_ * kSamplerBindingCount},
    };
    VkDescriptorPoolCreateInfo create{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    create.maxSets = sets_per_pool_;
    create.poolSizeCount = static_cast<uint32_t>(sizes.size());
    create.pPoolSizes = sizes.data();
    VkDescriptorPool pool = VK_NULL_HANDLE;
    if (!Result(create_pool_(device_, &create, nullptr, &pool), "vkCreateDescriptorPool", error)) return false;
    pools_.push_back({pool, 0});
    return true;
}

bool DescriptorArena::ValidateResources(const DescriptorResources& resources, std::string& error) const {
    if (!layout_ || !device_) {
        error = "Vulkan descriptor arena is not initialized";
        return false;
    }
    if (!resources.constant_buffer || !resources.constant_range || resources.constant_range > max_uniform_range_ ||
        resources.constant_range > resources.constant_buffer_size) {
        error = "Vulkan descriptor constants have an invalid buffer or range";
        return false;
    }
    if (resources.texture_layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL &&
        resources.texture_layout != VK_IMAGE_LAYOUT_GENERAL) {
        error = "Vulkan sampled-image descriptor has an unsupported layout";
        return false;
    }
    if (std::any_of(resources.textures.begin(), resources.textures.end(), [](VkImageView view) { return !view; }) ||
        std::any_of(resources.samplers.begin(), resources.samplers.end(), [](VkSampler sampler) { return !sampler; })) {
        error = "Vulkan descriptor resources require all fixed texture and sampler slots";
        return false;
    }
    return true;
}

bool DescriptorArena::Allocate(const DescriptorResources& resources, DescriptorSet& set, std::string& error) {
    set = {};
    error.clear();
    if (!ValidateResources(resources, error)) return false;
    if (in_flight_batch_) {
        error = "Vulkan descriptor arena cannot allocate while its owner batch is pending";
        return false;
    }
    VkDescriptorSet descriptor = VK_NULL_HANDLE;
    for (;;) {
        if (pool_at_ == pools_.size() && !CreatePool(error)) return false;
        Pool& pool = pools_[pool_at_];
        if (pool.allocated >= sets_per_pool_) {
            ++pool_at_;
            continue;
        }
        VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocate.descriptorPool = pool.handle;
        allocate.descriptorSetCount = 1;
        allocate.pSetLayouts = &layout_;
        const VkResult result = allocate_sets_(device_, &allocate, &descriptor);
        if (result == VK_SUCCESS) {
            ++pool.allocated;
            break;
        }
        if (result != VK_ERROR_OUT_OF_POOL_MEMORY && result != VK_ERROR_FRAGMENTED_POOL) {
            return Result(result, "vkAllocateDescriptorSets", error);
        }
        pool.allocated = sets_per_pool_;
        ++pool_at_;
    }

    std::array<VkDescriptorBufferInfo, kDynamicConstantBindingCount> buffers{};
    std::array<VkDescriptorImageInfo, kTextureBindingCount> images{};
    std::array<VkDescriptorImageInfo, kSamplerBindingCount> samplers{};
    std::array<VkWriteDescriptorSet, kDynamicConstantBindingCount + kTextureBindingCount + kSamplerBindingCount> writes{};
    uint32_t write_at = 0;
    for (uint32_t i = 0; i < kDynamicConstantBindingCount; ++i) {
        buffers[i] = {resources.constant_buffer, 0, resources.constant_range};
        writes[write_at++] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor, i, 0, 1,
                              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, nullptr, &buffers[i], nullptr};
    }
    for (uint32_t i = 0; i < kTextureBindingCount; ++i) {
        images[i] = {VK_NULL_HANDLE, resources.textures[i], resources.texture_layout};
        writes[write_at++] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor, kTextureBindingBase + i, 0, 1,
                              VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &images[i], nullptr, nullptr};
    }
    for (uint32_t i = 0; i < kSamplerBindingCount; ++i) {
        samplers[i] = {resources.samplers[i], VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
        writes[write_at++] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor, kSamplerBindingBase + i, 0, 1,
                              VK_DESCRIPTOR_TYPE_SAMPLER, &samplers[i], nullptr, nullptr};
    }
    update_sets_(device_, write_at, writes.data(), 0, nullptr);
    set = {descriptor, generation_, this, resources.constant_buffer_size, resources.constant_range};
    return true;
}

bool DescriptorArena::IsCurrent(const DescriptorSet& set) const {
    return set.owner == this && set.handle != VK_NULL_HANDLE && set.generation == generation_ &&
           in_flight_batch_ == nullptr;
}

bool DescriptorArena::Validate(const DescriptorSet& set,
                               const std::array<uint32_t, kDynamicConstantBindingCount>& dynamic_offsets,
                               std::string& error) const {
    error.clear();
    if (!IsCurrent(set) || !set.constant_range || !set.constant_buffer_size) {
        error = "Vulkan descriptor set is stale, unowned or has no constant-buffer range";
        return false;
    }
    for (uint32_t offset : dynamic_offsets) {
        if (offset % alignment_ != 0 || VkDeviceSize(offset) + set.constant_range > set.constant_buffer_size) {
            error = "Vulkan dynamic uniform offset is misaligned or outside its buffer range";
            return false;
        }
    }
    return true;
}

bool DescriptorArena::Bind(VkCommandBuffer command, VkPipelineLayout pipeline_layout, const DescriptorSet& set,
                           const std::array<uint32_t, kDynamicConstantBindingCount>& dynamic_offsets,
                           std::string& error) const {
    error.clear();
    if (!command || !pipeline_layout) {
        error = "Vulkan descriptor bind requires a command buffer and pipeline layout";
        return false;
    }
    if (!Validate(set, dynamic_offsets, error)) return false;
    bind_sets_(command, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, kDescriptorSetIndex, 1, &set.handle,
               kDynamicConstantBindingCount, dynamic_offsets.data());
    return true;
}

bool DescriptorArena::MarkInFlight(const CommandBatch& batch, std::string& error) {
    error.clear();
    if (in_flight_batch_) {
        error = "Vulkan descriptor arena already has an in-flight owner";
        return false;
    }
    if (!batch.Pending()) {
        error = "Vulkan descriptor arena owner batch is not pending after submission";
        return false;
    }
    in_flight_batch_ = &batch;
    return true;
}

bool DescriptorArena::ResetPools(std::string& error) {
    if (++generation_ == 0) generation_ = 1;
    for (Pool& pool : pools_) {
        if (!Result(reset_pool_(device_, pool.handle, 0), "vkResetDescriptorPool", error)) {
            pool_at_ = pools_.size();
            in_flight_batch_ = nullptr;
            return false;
        }
        pool.allocated = 0;
    }
    pool_at_ = 0;
    in_flight_batch_ = nullptr;
    return true;
}

bool DescriptorArena::ResetAfterCompletion(CommandBatch& batch, uint64_t timeout_ns, std::string& error) {
    error.clear();
    if (!device_ || !reset_pool_) {
        error = "Vulkan descriptor arena is not initialized";
        return false;
    }
    if (in_flight_batch_ && in_flight_batch_ != &batch) {
        error = "Vulkan descriptor arena completion owner does not match the pending batch";
        return false;
    }
    if (!batch.Wait(timeout_ns, error)) return false;
    return ResetPools(error);
}

}  // namespace gpu::vulkan
