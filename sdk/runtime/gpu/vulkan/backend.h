#pragma once

#include "gpu/backend.h"
#include "core/diag.h"
#include "gpu/shaders.h"
#include "gpu/texture.h"
#include "gpu/vertex.h"
#include "gpu/vulkan/buffer.h"
#include "gpu/vulkan/blit.h"
#include "gpu/vulkan/commands.h"
#include "gpu/vulkan/context.h"
#include "gpu/vulkan/descriptors.h"
#include "gpu/vulkan/draw_dispatch.h"
#include "gpu/vulkan/image.h"
#include "gpu/vulkan/late_readback.h"
#include "gpu/vulkan/pipeline_state.h"
#include "gpu/vulkan/shader.h"
#include "gpu/vulkan/shader_cache.h"
#include "gpu/vulkan/sampler.h"
#include "gpu/vulkan/shader_sources.h"
#include "gpu/vulkan/swapchain.h"
#include "gpu/vulkan/upload_allocator.h"
#include "gpu/vulkan/win32_surface.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace gpu::remaster {
struct Image;
}

namespace gpu {

class VulkanBackend final : public Backend, public UiServices {
public:
    VulkanBackend() = default;
    ~VulkanBackend() override;

    bool Init(uint32_t width, uint32_t height) override;
    bool Failed() const override { return terminal_.load(std::memory_order_acquire); }
    void Clear(const State&, uint32_t count, const uint8_t* rects, uint32_t flags, uint32_t color, float z,
               uint32_t stencil) override;
    void Draw(const State&, uint32_t prim, uint32_t start, uint32_t count, const uint8_t* indices) override;
    void Copy(const Surface& src, const Surface& dst, uint32_t count, uint32_t src_rects_va,
              uint32_t dst_points_va) override;
    void Present(const Surface& front) override;
    void Sync(uint32_t data_va, uint32_t size) override;
    void SyncLate(uint32_t data_va) override;
    bool DeliverLate(uint32_t data_va) override;
    void RecordLate(uint32_t data_va) override;
    void Flush() override;
    void PrepareShutdown() override;
    void SetGamma(const uint8_t* ramp) override;
    uint64_t UiImage(const std::string& key, const remaster::Image* image, bool source_missing) override;
    bool TvScreen(float[4]) override { return false; }

private:
    struct Target {
        std::unique_ptr<vulkan::Image> image;
        Surface surface{};
        uint32_t guest_width = 0, guest_height = 0;
        uint32_t width = 0, height = 0;
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
        bool depth = false;
        bool drawn = false;
        uint64_t drawn_generation = 0;
        uint64_t content_generation = 0;
        uint64_t content_epoch = 0;
        uint64_t content_signature = 0;
        std::vector<uint8_t> readback;
    };
    struct Texture {
        std::unique_ptr<vulkan::Image> image;
        TextureDesc desc{};
        uint64_t id = 0;
        uint64_t epoch = 0;
        uint64_t content_generation = 0;
        uint64_t content_signature = 0;
        uint32_t palette_va = 0;
        size_t palette_bytes = 0;
        uint64_t palette_hash = 0;
        uint64_t used_generation = 0;
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    };
    struct PipelineKey {
        uint64_t vertex = 0, pixel = 0, geometry = 0;
        uint16_t regs = 0;
        uint8_t topology = 0, strip = 0;
        VkFormat color = VK_FORMAT_UNDEFINED, depth = VK_FORMAT_UNDEFINED;
        std::array<uint32_t, 16> elements{};
        std::array<uint32_t, 16> strides{};
        uint32_t buffers = 0;
        uint32_t blend = 0, src = 0, dst = 0, op = 0, write_mask = 0;
        uint32_t depth_enable = 0, depth_write = 0, depth_func = 0;
        uint32_t cull = 0, front_ccw = 0, fill = 0;
        uint32_t stencil_enable = 0, stencil_func = 0, stencil_mask = 0, stencil_write_mask = 0;
        uint32_t stencil_fail = 0, stencil_zfail = 0, stencil_pass = 0;
        int32_t depth_bias = 0;
        uint32_t depth_slope_bits = 0;
        bool operator==(const PipelineKey&) const = default;
    };
    struct PipelineHash {
        size_t operator()(const PipelineKey& key) const {
            size_t hash = 0;
            auto add = [&](uint64_t value) {
                hash ^= std::hash<uint64_t>{}(value) + size_t(0x9e3779b9) + (hash << 6) + (hash >> 2);
            };
            add(key.vertex); add(key.pixel); add(key.geometry); add(key.regs);
            add(key.topology); add(key.strip); add(key.color); add(key.depth);
            for (auto value : key.elements) add(value);
            for (auto value : key.strides) add(value);
            add(key.buffers); add(key.blend); add(key.src); add(key.dst); add(key.op); add(key.write_mask);
            add(key.depth_enable); add(key.depth_write); add(key.depth_func);
            add(key.cull); add(key.front_ccw); add(key.fill);
            add(key.stencil_enable); add(key.stencil_func); add(key.stencil_mask); add(key.stencil_write_mask);
            add(key.stencil_fail); add(key.stencil_zfail); add(key.stencil_pass);
            add(static_cast<uint32_t>(key.depth_bias)); add(key.depth_slope_bits);
            return hash;
        }
    };
    struct SamplerKey {
        uint32_t mag = 0, min = 0, mip = 0;
        uint32_t address_u = 0, address_v = 0, address_w = 0;
        uint32_t mip_lod_bias = 0, max_mip_level = 0, max_anisotropy = 0, border_color = 0;
        uint32_t compare_enable = 0, compare_op = VK_COMPARE_OP_ALWAYS;
        VkFormat sampled_format = VK_FORMAT_UNDEFINED;
        bool operator==(const SamplerKey&) const = default;
    };
    struct SamplerHash {
        size_t operator()(const SamplerKey& key) const {
            size_t hash = 0;
            auto add = [&](uint32_t value) {
                hash ^= std::hash<uint32_t>{}(value) + size_t(0x9e3779b9) + (hash << 6) + (hash >> 2);
            };
            add(key.mag); add(key.min); add(key.mip);
            add(key.address_u); add(key.address_v); add(key.address_w);
            add(key.mip_lod_bias); add(key.max_mip_level); add(key.max_anisotropy); add(key.border_color);
            add(key.compare_enable); add(key.compare_op);
            add(key.sampled_format);
            return hash;
        }
    };

    struct FrameContext {
        vulkan::CommandBatch batch;
        vulkan::DescriptorArena descriptors;
        VkCommandBuffer command = VK_NULL_HANDLE;
        std::vector<std::unique_ptr<vulkan::Buffer>> frame_buffers;
        vulkan::UploadAllocator uploads;
        // Images replaced while recording must remain alive until this batch retires.
        std::vector<std::unique_ptr<vulkan::Image>> frame_images;
        bool recording = false;
        bool descriptors_used = false;
        uint64_t submission_ordinal = 0;
        VkFence completion_fence = VK_NULL_HANDLE;

        VkQueryPool perf_query_pool = VK_NULL_HANDLE;
        bool perf_detail_frame_active = false;
        bool perf_frame_active = false;
        bool perf_frame_ended = false;
        bool perf_query_pending = false;
        uint64_t perf_current_draws = 0;
        uint64_t perf_current_pipeline_misses = 0;
        uint64_t perf_current_shader_modules = 0;
        uint64_t perf_current_source_hits = 0;
        uint64_t perf_current_source_misses = 0;
        uint64_t perf_current_texture_uploads = 0;
        uint64_t perf_current_descriptor_requests = 0;
        uint64_t perf_current_record_ns = 0;
        uint64_t perf_current_wait_ns = 0;
        uint64_t perf_current_detail_vertex_assembly_ns = 0;
        uint64_t perf_current_detail_texture_target_sampler_ns = 0;
        uint64_t perf_current_detail_pipeline_source_ns = 0;
        uint64_t perf_current_detail_constant_vertex_index_upload_ns = 0;
        uint64_t perf_current_detail_descriptor_lookup_allocation_ns = 0;
        uint64_t perf_current_detail_command_recording_ns = 0;
        uint64_t perf_current_detail_swapchain_render_wait_ns = 0;
        uint64_t perf_current_detail_batch_completion_wait_ns = 0;
        uint64_t perf_current_detail_late_publish_ns = 0;
        uint64_t perf_current_detail_descriptor_retire_ns = 0;
        uint64_t perf_current_detail_resource_retire_ns = 0;
        uint64_t perf_current_upload_allocations = 0;
        uint64_t perf_current_upload_bytes = 0;
        uint64_t perf_current_upload_block_grows = 0;
        bool perf_current_gpu_valid = false;
        double perf_current_gpu_ms = 0.0;
        std::chrono::steady_clock::time_point perf_frame_start;
        double perf_current_frame_ms = 0.0;
    };

    bool BeginFrame();
    bool CompleteFrame();
    bool RetireFrameContext(uint32_t index, uint64_t timeout_ns, const char* operation);
    bool DrainFrameContexts();
    FrameContext& ActiveFrame() { return *active_frame_; }
    const FrameContext& ActiveFrame() const { return *active_frame_; }
    bool Fail(const char* operation, const std::string& detail = {});
    bool EnsureTarget(const Surface& surface, bool depth, Target*& out);
    bool EnsureTexture(const State& state, int stage, Texture*& out, TextureDesc& desc);
    bool CompatibleTarget(const Target& target, const TextureDesc& desc, bool depth) const;
    bool PrepareTargetForSampling(Target& target);
    bool SnapshotTargetForSampling(Target& target, VkImageView& view);
    bool LatchCpuWrite(Target& target);
    bool RecordTargetCopy(Target& source, Target& destination, uint32_t count, uint32_t src_rects,
                          uint32_t dst_points);
    bool UploadGuestImage(vulkan::Image& image, VkImageLayout& layout, const TextureDesc& desc,
                          const uint32_t* palette);
    bool UploadSurface(Target& target);
    bool CreateDummy();
    bool CreateSamplers();
    bool CreatePipelineLayout();
    bool CreateBlitPipeline();
    void UpdateGammaRamp();
    bool CreateUi();
    bool RecreateUi();
    bool PumpOverrides();
    bool FlushOverrideUploads();
    bool RecordSwapchainScreenshot(VkImage image, VkImageLayout& layout);
    bool FinishSwapchainScreenshot();
    bool RenderUi(const vulkan::AcquireResult& acquired);
    bool FlushUiUploads(const vulkan::AcquireResult& acquired);
    bool InitPerfQueries(uint32_t forced_interval = 0);
    bool LoadDrawDispatch();
    void ClearDrawDispatch();
    void ReportCapabilities();
    void ResetPerfQueries();
    void BeginPerfFrame();
    void EndPerfFrame();
    void FinishPerfFrame(uint64_t wait_ns);
    void ReportPerfWindow();
    const vulkan::DrawSources& DrawSourcesFor(const State& state, const Assembled& assembled,
                                              const PixelKey& pixel_key);
    size_t UploadBlockIndex(const vulkan::Buffer* buffer) const;
    bool CreateDrawPipeline(const State& state, const Assembled& assembled, const std::string& vs_source,
                            const std::string& ps_source, const std::string& gs_source, VkFormat color_format,
                            VkFormat depth_format, VkPipeline& pipeline, PipelineKey& key);
    bool CompileModule(const std::string& source, const char* entry, vulkan::ShaderStage stage,
                       vulkan::ShaderModule& module, VkShaderModule& shader);
    bool AllocateUpload(VkDeviceSize size, VkDeviceSize alignment, vulkan::Buffer*& buffer, VkDeviceSize& offset);
    void RetireUploads();
    bool UploadDrawConstants(const State& state, const DrawConstants& constants, vulkan::Buffer*& buffer,
                             std::array<uint32_t, vulkan::kDynamicConstantBindingCount>& offsets);
    bool AllocateDescriptors(const std::array<VkImageView, vulkan::kTextureBindingCount>& textures,
                             const std::array<VkSampler, vulkan::kSamplerBindingCount>& samplers,
                             vulkan::Buffer& constants, VkDeviceSize range, vulkan::DescriptorSet& set);
    bool BeginRendering(Target* color, Target* depth, VkImageView color_view = VK_NULL_HANDLE,
                        VkFormat color_format = VK_FORMAT_UNDEFINED, VkExtent2D extent = {});
    void EndRendering();
    void Transition(Target& target, VkImageLayout layout);
    void Transition(vulkan::Image& image, VkImageLayout& current, VkImageLayout layout, VkImageAspectFlags aspect);
    void Transition(VkImage image, VkImageAspectFlags aspect, VkImageLayout& current, VkImageLayout layout);
    void TransitionSwapchainColorAttachment(VkImage image, VkImageLayout& current);
    bool RecordDraw(const State& state, const Assembled& assembled, const PixelKey& pixel_key, Target* color, Target* depth,
                    const std::array<VkImageView, 4>& stage_views, const std::array<VkSampler, 4>& stage_samplers,
                    const DrawConstants& constants);
    bool RecordBlit(Target* source, const vulkan::AcquireResult& acquired);
    bool Rects(const State& state, Target& target, uint32_t count, const uint8_t* rects, std::vector<VkClearRect>& out);
    bool ReadBackPixels(Target& target, std::vector<uint8_t>& pixels);
    bool ReadBack(Target& target, uint32_t data_va);
    void StampLateSubmissions();
    void PublishLateCompletion();

    VkSampler SamplerFor(const State& state, int stage, bool shadow, VkFormat sampled_format);
    VkPrimitiveTopology Topology(const Assembled& assembled) const;

    vulkan::Context context_;
    vulkan::Win32Surface surface_;
    vulkan::Swapchain swapchain_;
    static constexpr uint32_t kFrameContextCount = 2;
    std::array<FrameContext, kFrameContextCount> frame_contexts_;
    FrameContext* active_frame_ = &frame_contexts_[0];
    uint32_t active_frame_index_ = 0;
    bool frame_started_ = false;
    uint64_t next_submission_ordinal_ = 0;
    uint64_t completed_submission_ordinal_ = 0;
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline blit_pipeline_ = VK_NULL_HANDLE;
    VkSampler linear_sampler_ = VK_NULL_HANDLE;
    VkSampler point_sampler_ = VK_NULL_HANDLE;
    std::unique_ptr<vulkan::Image> dummy_;
    VkImageLayout dummy_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    std::unordered_map<uint64_t, Target> targets_;
    std::unordered_map<uint64_t, Texture> textures_;
    std::unordered_map<PipelineKey, VkPipeline, PipelineHash> pipelines_;
    vulkan::ShaderCache shader_cache_;
    vulkan::DrawSourceCache shader_sources_;
    struct OverrideTex {
        std::unique_ptr<vulkan::Image> image;
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    };
    struct PendingOverrideUpload {
        uint64_t id = 0;
        std::unique_ptr<vulkan::Buffer> staging;
        uint32_t width = 0, height = 0;
    };
    std::unordered_map<uint64_t, OverrideTex> override_textures_;
    std::vector<PendingOverrideUpload> pending_override_uploads_;
    std::vector<VkSampler> samplers_;
    std::unordered_map<SamplerKey, VkSampler, SamplerHash> stage_samplers_;
    uint32_t custom_border_sampler_count_ = 0;
    std::unordered_map<VkImage, VkImageLayout> external_layouts_;

    struct LateSlot {
        std::unique_ptr<vulkan::Buffer> readback;
        TextureDesc desc{};
        uint32_t width = 0, height = 0;
        VkImage image_identity = VK_NULL_HANDLE;
    };
    struct Late {
        static constexpr uint32_t kSlotCount = 3;
        std::array<LateSlot, kSlotCount> slots;
        std::array<vulkan::LateReadbackSlotState, kSlotCount> states;
        TextureDesc desc{};
        uint32_t next = 0;
        uint32_t width = 0, height = 0;
        uint64_t identity_generation = 0;
        VkImage image_identity = VK_NULL_HANDLE;
        uint64_t acknowledged_epoch = 0;
        uint64_t acknowledged_signature = 0;
        uint64_t acknowledged_identity = 0;
        bool acknowledgement_pending = false;
        uint64_t delivered_serial = 0;
        bool configured = false;
        bool retire_pending = false;
    };
    std::unordered_map<uint32_t, Late> late_;
    std::mutex late_lock_;
    std::atomic<uint64_t> late_published_generation_{0};
    std::atomic<uint64_t> late_published_completed_serial_{0};
    std::atomic<int> late_published_error_{VK_SUCCESS};
    std::string last_error_;
    uint32_t guest_width_ = 0, guest_height_ = 0;
    float render_scale_ = 1.0f;
    bool rendering_ = false;
    std::atomic<bool> terminal_{false};
    bool capability_reported_ = false;
    uint64_t capability_settings_signature_ = 0;
    Target* rendering_color_ = nullptr;
    Target* rendering_depth_ = nullptr;
    bool recolor_ = false;
    uint32_t recolor_argb_ = 0;
    VertexElement recolor_element_{};
    uint64_t frame_count_ = 0;
    // Bit mask for the distinct fog-state variants seen while draw tracing is armed.
    uint64_t fog_trace_seen_ = 0;
    bool sky_phase_ = true, sky_scene_done_ = false;
    Target* sky_scene_color_ = nullptr;
    Target* sky_scene_depth_ = nullptr;
    uint32_t sky_layer_ = 0;
    uint64_t frame_generation_ = 0;
    bool gamma_dirty_ = true;
    bool gamma_effective_ = false;
    uint32_t brightness_ = 0;
    uint8_t gamma_[768] = {};
    uint8_t gamma_lut_[768] = {};
    vulkan::DrawDispatch draw_dispatch_;

    struct UiTex {
        std::unique_ptr<vulkan::Image> image;
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkDescriptorSet descriptor = VK_NULL_HANDLE;
    };
    struct PendingUiUpload {
        std::string key;
        std::unique_ptr<vulkan::Buffer> staging;
        VkFormat format = VK_FORMAT_UNDEFINED;
        uint32_t width = 0, height = 0;
    };
    VkDescriptorPool ui_pool_ = VK_NULL_HANDLE;
    VkFormat ui_color_format_ = VK_FORMAT_UNDEFINED;
    std::unordered_map<std::string, UiTex> ui_textures_;
    std::vector<PendingUiUpload> pending_ui_uploads_;
    bool ui_initialized_ = false;
    std::wstring pending_screenshot_;
    std::unique_ptr<vulkan::Buffer> screenshot_buffer_;
    uint32_t screenshot_width_ = 0, screenshot_height_ = 0;
    double fps_ = 0;
    uint64_t fps_sample_frame_ = 0;
    std::chrono::steady_clock::time_point fps_sample_time_;

    // Optional bounded diagnostics. They are enabled only with XBE_VK_PERF and have no
    // effect on normal rendering when the variable is absent.
    bool perf_enabled_ = false;
    // Detail timing is opt-in and only records while the existing perf trace control is armed.
    bool perf_detail_enabled_ = false;
    uint32_t perf_interval_ = 0;
    uint32_t perf_upload_trace_count_ = 0;
    uint64_t perf_window_frames_ = 0;
    uint64_t perf_window_draws_ = 0;
    uint64_t perf_window_pipeline_misses_ = 0;
    uint64_t perf_window_shader_modules_ = 0;
    uint64_t perf_window_source_hits_ = 0;
    uint64_t perf_window_source_misses_ = 0;
    uint64_t perf_window_texture_uploads_ = 0;
    uint64_t perf_window_descriptor_requests_ = 0;
    uint64_t perf_window_record_ns_ = 0;
    uint64_t perf_window_wait_ns_ = 0;
    uint64_t perf_window_upload_allocations_ = 0;
    uint64_t perf_window_upload_bytes_ = 0;
    uint64_t perf_window_upload_block_grows_ = 0;
    uint64_t perf_window_upload_rewinds_ = 0;
    size_t perf_window_peak_upload_blocks_ = 0;
    uint64_t perf_window_gpu_samples_ = 0;
    double perf_window_gpu_ms_ = 0.0;
    // Late-readback coverage is kept in the existing opt-in perf window. DeliverLate can
    // run on the title thread while the render thread reports and resets the window.
    std::atomic<bool> perf_counters_enabled_{false};
    std::atomic<uint64_t> perf_window_sync_requests_{0};
    std::atomic<uint64_t> perf_window_deliver_late_hits_{0};
    std::atomic<uint64_t> perf_window_deliver_late_misses_{0};
    std::atomic<uint64_t> perf_window_record_late_accepted_{0};
    std::atomic<uint64_t> perf_window_record_late_skips_{0};
    std::atomic<uint64_t> perf_window_guest_writeback_successes_{0};
    diag::PerfTrace perf_trace_{"vulkan"};
    uint64_t perf_trace_frame_index_ = 0;
};

std::unique_ptr<Backend> CreateVulkanBackend();

}  // namespace gpu
