#pragma once

#include "gpu/vulkan/descriptors.h"
#include "gpu/vulkan/shader.h"

#include <array>
#include <cstdint>
#include <string>

namespace gpu::vulkan {

enum class GraphicsPass : uint8_t {
    SampledImage,
};

struct GraphicsPassContract {
    GraphicsPass pass = GraphicsPass::SampledImage;
    std::string name = "sampled-image";
    VkShaderStageFlags stages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    uint32_t color_attachment_count = 1;
    bool has_depth = false;
};

struct GraphicsPipelineDesc {
    GraphicsPassContract contract{};
    const ShaderModule* vertex = nullptr;
    const ShaderModule* fragment = nullptr;
    const ShaderModule* geometry = nullptr;
    const ShaderModule* hull = nullptr;
    const ShaderModule* domain = nullptr;
    VkFormat color_format = VK_FORMAT_UNDEFINED;
    VkFormat depth_format = VK_FORMAT_UNDEFINED;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
    VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
};

struct RenderTarget {
    VkImageView color = VK_NULL_HANDLE;
    VkFormat color_format = VK_FORMAT_UNDEFINED;
    VkExtent2D extent{};
    VkClearValue clear{};
};

// A reusable graphics pipeline for an explicit pass contract. The initial
// contract records a VS/PS sampled-image draw into one dynamic-rendering color
// target; later passes can add their own contract and attachment requirements.
class GraphicsPipeline {
public:
    GraphicsPipeline() = default;
    GraphicsPipeline(const GraphicsPipeline&) = delete;
    GraphicsPipeline& operator=(const GraphicsPipeline&) = delete;
    ~GraphicsPipeline();

    bool Init(const Context& context, const DescriptorArena& descriptors, const GraphicsPipelineDesc& description,
              std::string& error);
    void Reset();

    bool RecordDraw(VkCommandBuffer command, const DescriptorArena& descriptors, const DescriptorSet& set,
                    const std::array<uint32_t, kDynamicConstantBindingCount>& dynamic_offsets,
                    const RenderTarget& target, uint32_t vertex_count, std::string& error) const;

    VkPipeline Handle() const { return pipeline_; }
    VkPipelineLayout Layout() const { return layout_; }
    GraphicsPass Pass() const { return contract_.pass; }

private:
    bool ValidateShader(const ShaderModule* module, ShaderStage expected, const char* name, std::string& error) const;

    VkDevice device_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    GraphicsPassContract contract_{};
    VkFormat color_format_ = VK_FORMAT_UNDEFINED;
    VkFormat depth_format_ = VK_FORMAT_UNDEFINED;
    VkSampleCountFlagBits samples_ = VK_SAMPLE_COUNT_1_BIT;
    PFN_vkDestroyPipeline destroy_pipeline_ = nullptr;
    PFN_vkDestroyPipelineLayout destroy_layout_ = nullptr;
    PFN_vkDestroyShaderModule destroy_shader_module_ = nullptr;
    PFN_vkCmdBeginRendering begin_rendering_ = nullptr;
    PFN_vkCmdEndRendering end_rendering_ = nullptr;
    PFN_vkCmdBindPipeline bind_pipeline_ = nullptr;
    PFN_vkCmdSetViewport set_viewport_ = nullptr;
    PFN_vkCmdSetScissor set_scissor_ = nullptr;
    PFN_vkCmdDraw draw_ = nullptr;
};

}  // namespace gpu::vulkan
