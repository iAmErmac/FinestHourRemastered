#pragma once

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <string>
#include <type_traits>

namespace gpu::vulkan {

// Device functions used by the hot draw/rendering path. The resolver is called only during
// initialization; callers must clear this table before the owning VkDevice is destroyed.
struct DrawDispatch {
    PFN_vkCmdBindPipeline bind_pipeline = nullptr;
    PFN_vkCmdBindVertexBuffers bind_vertex_buffers = nullptr;
    PFN_vkCmdBindIndexBuffer bind_index_buffer = nullptr;
    PFN_vkCmdDraw draw = nullptr;
    PFN_vkCmdDrawIndexed draw_indexed = nullptr;
    PFN_vkCmdSetViewport set_viewport = nullptr;
    PFN_vkCmdSetScissor set_scissor = nullptr;
    PFN_vkCmdSetBlendConstants set_blend_constants = nullptr;
    PFN_vkCmdSetStencilReference set_stencil_reference = nullptr;
    PFN_vkCmdBeginRendering begin_rendering = nullptr;
    PFN_vkCmdEndRendering end_rendering = nullptr;
    PFN_vkCmdPipelineBarrier pipeline_barrier = nullptr;

    void Clear() { *this = {}; }

    template <typename Resolver>
    bool Load(Resolver&& resolve, std::string& error) {
        const char* missing = nullptr;
        auto required = [&](auto& slot, const char* name) {
            slot = reinterpret_cast<std::decay_t<decltype(slot)>>(resolve(name));
            if (!slot && !missing) missing = name;
        };
        required(bind_pipeline, "vkCmdBindPipeline");
        required(bind_vertex_buffers, "vkCmdBindVertexBuffers");
        required(bind_index_buffer, "vkCmdBindIndexBuffer");
        required(draw, "vkCmdDraw");
        required(draw_indexed, "vkCmdDrawIndexed");
        required(set_viewport, "vkCmdSetViewport");
        required(set_scissor, "vkCmdSetScissor");
        required(set_blend_constants, "vkCmdSetBlendConstants");
        required(set_stencil_reference, "vkCmdSetStencilReference");
        required(begin_rendering, "vkCmdBeginRendering");
        required(end_rendering, "vkCmdEndRendering");
        required(pipeline_barrier, "vkCmdPipelineBarrier");
        if (!missing) return true;
        Clear();
        error = std::string("missing Vulkan draw entry point: ") + missing;
        return false;
    }
};

}  // namespace gpu::vulkan
