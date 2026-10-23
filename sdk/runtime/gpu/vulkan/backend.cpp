#include "gpu/vulkan/backend.h"
#include "gpu/vulkan/copy.h"
#include "gpu/vulkan/raster.h"

#include <imgui.h>
#include <backends/imgui_impl_vulkan.h>
#include <backends/imgui_impl_win32.h>

#include "app/app.h"
#include "core/control.h"
#include "core/log.h"
#include "gpu/shaders.h"
#include "gpu/draw_constants.h"
#include "gpu/gamma.h"
#include "gpu/remaster.h"
#include "gpu/texture.h"
#include "gpu/vertex.h"
#include "kernel/kernel.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <string>
#include <dxgiformat.h>

namespace gpu {
namespace {

constexpr VkFormat kColorFormat = VK_FORMAT_B8G8R8A8_UNORM;
constexpr VkFormat kDepthFormat = VK_FORMAT_D24_UNORM_S8_UINT;
constexpr VkDeviceSize kConstantAlignment = 256;
constexpr VkDeviceSize kConstantRange = 16ull * 1024;
constexpr float kDepth24Max = 16777215.0f;

VkFormat UiFormat(DXGI_FORMAT format) {
    switch (format) {
        case DXGI_FORMAT_R8G8B8A8_UNORM: return VK_FORMAT_R8G8B8A8_UNORM;
        case DXGI_FORMAT_B8G8R8A8_UNORM: return VK_FORMAT_B8G8R8A8_UNORM;
        case DXGI_FORMAT_BC1_UNORM: return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
        case DXGI_FORMAT_BC2_UNORM: return VK_FORMAT_BC2_UNORM_BLOCK;
        case DXGI_FORMAT_BC3_UNORM: return VK_FORMAT_BC3_UNORM_BLOCK;
        case DXGI_FORMAT_BC7_UNORM: return VK_FORMAT_BC7_UNORM_BLOCK;
        default: return VK_FORMAT_UNDEFINED;
    }
}

PFN_vkVoidFunction ImGuiVulkanFunction(const char* name, void* user_data) {
    auto* context = static_cast<vulkan::Context*>(user_data);
    if (auto function = context->DeviceProc(name)) return function;
    return context->InstanceProc(name);
}

template <typename T>
T DeviceProc(const vulkan::Context& context, const char* name) {
    return reinterpret_cast<T>(context.DeviceProc(name));
}

template <typename T>
T InstanceProc(const vulkan::Context& context, const char* name) {
    return reinterpret_cast<T>(context.InstanceProc(name));
}

bool Result(VkResult result, const char* operation, std::string& error) {
    if (result == VK_SUCCESS) return true;
    error = std::string(operation) + " failed: VkResult " + std::to_string(static_cast<int>(result));
    return false;
}

uint64_t ElapsedNanoseconds(std::chrono::steady_clock::time_point start) {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start).count());
}

float Bits(uint32_t value) {
    float result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

void Color(uint32_t argb, float* result) {
    result[0] = float((argb >> 16) & 255) / 255.0f;
    result[1] = float((argb >> 8) & 255) / 255.0f;
    result[2] = float(argb & 255) / 255.0f;
    result[3] = float((argb >> 24) & 255) / 255.0f;
}

uint32_t SimpleState(const State& state, uint32_t method) {
    return state.simple[method >> 2];
}

VkImageAspectFlags ColorAspect() { return VK_IMAGE_ASPECT_COLOR_BIT; }

VkImageAspectFlags DepthAspect() { return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT; }

VkDeviceSize Align(VkDeviceSize value, VkDeviceSize alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

void LayoutMasks(VkImageLayout layout, VkPipelineStageFlags& stages, VkAccessFlags& access) {
    switch (layout) {
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
            access = VK_ACCESS_TRANSFER_WRITE_BIT;
            break;
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
            access = VK_ACCESS_TRANSFER_READ_BIT;
            break;
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            stages = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            access = VK_ACCESS_SHADER_READ_BIT;
            break;
        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            stages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            access = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            break;
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
            stages = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
            access = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            break;
        case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
            stages = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
            access = 0;
            break;
        default:
            stages = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            access = 0;
            break;
    }
}

void MatMul(const float* x, const float* y, float* out) {
    for (int row = 0; row < 4; ++row)
        for (int column = 0; column < 4; ++column)
            out[row * 4 + column] = x[row * 4] * y[column] + x[row * 4 + 1] * y[4 + column] +
                                    x[row * 4 + 2] * y[8 + column] + x[row * 4 + 3] * y[12 + column];
}

void Transpose(const float* matrix, float* out) {
    for (int row = 0; row < 4; ++row)
        for (int column = 0; column < 4; ++column) out[column * 4 + row] = matrix[row * 4 + column];
}

void InverseTranspose3x3(const float* wv, float* out) {
    float det = wv[0] * (wv[5] * wv[10] - wv[6] * wv[9]) - wv[1] * (wv[4] * wv[10] - wv[6] * wv[8]) +
                wv[2] * (wv[4] * wv[9] - wv[5] * wv[8]);
    float inverse = det ? 1.0f / det : 0.0f;
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 3; ++column) {
            int row1 = (row + 1) % 3, row2 = (row + 2) % 3;
            int col1 = (column + 1) % 3, col2 = (column + 2) % 3;
            out[row * 4 + column] =
                (wv[row1 * 4 + col1] * wv[row2 * 4 + col2] - wv[row1 * 4 + col2] * wv[row2 * 4 + col1]) * inverse;
        }
}

void FillTransformConstants(const State& state, uint32_t width, uint32_t height, DrawConstants& constants) {
    constexpr size_t kView = 2, kProjection = 3, kTexture0 = 16, kWorld = 256;
    float world_view[16], world_view_projection[16];
    MatMul(state.transforms[kWorld].data(), state.transforms[kView].data(), world_view);
    MatMul(world_view, state.transforms[kProjection].data(), world_view_projection);
    const Viewport& viewport = state.viewport;
    float sx = float(viewport.w) / float(width), sy = float(viewport.h) / float(height);
    float ox = (2.0f * viewport.x + viewport.w) / float(width) - 1.0f;
    float oy = 1.0f - (2.0f * viewport.y + viewport.h) / float(height);
    float viewport_matrix[16] = {sx, 0, 0, 0, 0, sy, 0, 0, 0, 0, viewport.max_z - viewport.min_z, 0,
                                 ox, oy, viewport.min_z, 1};
    float full[16];
    MatMul(world_view_projection, viewport_matrix, full);
    Transpose(full, constants.wvp);
    Transpose(world_view, constants.wv);
    float normal[16] = {};
    InverseTranspose3x3(world_view, normal);
    Transpose(normal, constants.wvn);
    for (int texture = 0; texture < 4; ++texture)
        Transpose(state.transforms[kTexture0 + size_t(texture)].data(), constants.tm[texture]);
}

void FillConstants(const State& state, uint32_t width, uint32_t height, bool vertex_program, bool pretransformed,
                   const FixedKey& fixed, const std::array<TexInfo, 4>& infos, DrawConstants& constants) {
    draw_constants::Surface(state, width, height, constants);
    draw_constants::PretransformedFog(state, pretransformed, constants);
    draw_constants::Fog(state, vertex_program, constants);
    draw_constants::Combiners(state, constants);
    for (int stage = 0; stage < 4; ++stage) {
        TextureDesc desc;
        if (state.textures[size_t(stage)] && Describe(state.textures[size_t(stage)], desc)) {
            constants.texsize[stage][0] = float(desc.width);
            constants.texsize[stage][1] = float(desc.height);
        }
        const auto& tss = state.tss[size_t(stage)];
        constants.bump[stage][0] = Bits(tss[kTssBumpEnvMat00]);
        constants.bump[stage][1] = Bits(tss[kTssBumpEnvMat01]);
        constants.bump[stage][2] = Bits(tss[kTssBumpEnvMat10]);
        constants.bump[stage][3] = Bits(tss[kTssBumpEnvMat11]);
        constants.bumpl[stage][0] = Bits(tss[kTssBumpEnvLScale]);
        constants.bumpl[stage][1] = Bits(tss[kTssBumpEnvLOffset]);
        (void)infos;
    }
    if (vertex_program || !pretransformed) {
        FillTransformConstants(state, width, height, constants);
        draw_constants::Lights(state, fixed, constants);
    }
}

uint64_t SurfaceKey(const Surface& surface, bool depth) {
    const uint32_t data = surface.data ? mem::VirtualFromPhysical(surface.data) : 0;
    return (uint64_t(data) << 1) | uint64_t(depth);
}

bool SameLateDesc(const TextureDesc& a, const TextureDesc& b) {
    return a.data == b.data && a.format == b.format && a.width == b.width && a.height == b.height &&
           a.depth == b.depth && a.levels == b.levels && a.pitch == b.pitch && a.cube == b.cube &&
           a.linear == b.linear;
}

void UnturnedBox(const float* x, const float* y, uint32_t n, Draw2D& draw) {
    float ux[4], uy[4];
    uint32_t unique = 0;
    for (uint32_t i = 0; i < n; ++i) {
        bool seen = false;
        for (uint32_t j = 0; j < unique; ++j)
            seen |= std::fabs(ux[j] - x[i]) < 0.01f && std::fabs(uy[j] - y[i]) < 0.01f;
        if (!seen) {
            if (unique == 4) return;
            ux[unique] = x[i], uy[unique] = y[i], ++unique;
        }
    }
    if (unique != 4) return;
    bool square = true;
    for (uint32_t i = 0; i < 4; ++i)
        square &= std::fabs(ux[i] - draw.x0) < 0.01f || std::fabs(ux[i] - draw.x1) < 0.01f;
    if (square) return;
    float distances[6];
    uint32_t at = 0;
    for (uint32_t i = 0; i < 4; ++i)
        for (uint32_t j = i + 1; j < 4; ++j) distances[at++] = std::hypot(ux[i] - ux[j], uy[i] - uy[j]);
    std::sort(distances, distances + 6);
    float half = (distances[0] + distances[1] + distances[2] + distances[3]) / 8.0f;
    float cx = (ux[0] + ux[1] + ux[2] + ux[3]) / 4.0f;
    float cy = (uy[0] + uy[1] + uy[2] + uy[3]) / 4.0f;
    draw.ux0 = cx - half;
    draw.ux1 = cx + half;
    draw.uy0 = cy - half;
    draw.uy1 = cy + half;
}

void BuildDraw2D(const State& state, const Assembled& assembled, bool programmable, uint32_t guest_width,
                 uint32_t guest_height, uint64_t texture, Draw2D& draw, VertexElement& color_element) {
    draw = {1e9f, 1e9f, -1e9f, -1e9f, guest_width, guest_height, 1.0f,
            assembled.pretransformed ? 0u : state.layout2d, false, 0, 0xFFFFFFFFu, texture};
    color_element = {};
    const VertexElement& position = assembled.elements[0];
    const bool position_float = position.format == DXGI_FORMAT_R32G32B32A32_FLOAT ||
                                position.format == DXGI_FORMAT_R32G32B32_FLOAT ||
                                (assembled.pretransformed && position.format == DXGI_FORMAT_R32G32_FLOAT);
    const VertexBuffer* buffer = (assembled.regs & 1) && position.buffer < assembled.buffer_count
                                     ? &assembled.buffers[position.buffer]
                                     : nullptr;
    const bool pixels = assembled.pretransformed || (state.layout2d & 0x100);
    float corners_x[6] = {}, corners_y[6] = {};
    uint32_t corner_count = 0;
    if (position_float && buffer && buffer->src && buffer->stride && (pixels || !programmable)) {
        float world_view[16], world_view_projection[16];
        if (!pixels) {
            MatMul(state.transforms[256].data(), state.transforms[2].data(), world_view);
            MatMul(world_view, state.transforms[3].data(), world_view_projection);
        }
        const Viewport& viewport = state.viewport;
        uint32_t vertex_count = assembled.vertex_count ? assembled.vertex_count : buffer->bytes / buffer->stride;
        vertex_count = std::min(vertex_count, buffer->bytes / buffer->stride);
        for (uint32_t i = 0; i < vertex_count; ++i) {
            float p[3] = {};
            std::memcpy(p, buffer->src + size_t(i) * buffer->stride + position.offset,
                        position.format == DXGI_FORMAT_R32G32_FLOAT ? 8 : 12);
            float x = p[0], y = p[1];
            if (!pixels) {
                float clip_x = p[0] * world_view_projection[0] + p[1] * world_view_projection[4] +
                               p[2] * world_view_projection[8] + world_view_projection[12];
                float clip_y = p[0] * world_view_projection[1] + p[1] * world_view_projection[5] +
                               p[2] * world_view_projection[9] + world_view_projection[13];
                float clip_w = p[0] * world_view_projection[3] + p[1] * world_view_projection[7] +
                               p[2] * world_view_projection[11] + world_view_projection[15];
                if (clip_w <= 1e-6f) continue;
                x = viewport.x + (clip_x / clip_w * 0.5f + 0.5f) * viewport.w;
                y = viewport.y + (0.5f - clip_y / clip_w * 0.5f) * viewport.h;
            }
            draw.x0 = std::min(draw.x0, x);
            draw.x1 = std::max(draw.x1, x);
            draw.y0 = std::min(draw.y0, y);
            draw.y1 = std::max(draw.y1, y);
            draw.bounded = true;
            if (vertex_count <= 6) corners_x[corner_count] = x, corners_y[corner_count++] = y;
        }
        UnturnedBox(corners_x, corners_y, corner_count, draw);
    }
    if (!draw.bounded) draw.x0 = 0, draw.y0 = 0, draw.x1 = float(guest_width), draw.y1 = float(guest_height);
    if (!(draw.ux1 > draw.ux0)) draw.ux0 = draw.x0, draw.uy0 = draw.y0, draw.ux1 = draw.x1, draw.uy1 = draw.y1;
    if (assembled.regs & 8) {
        color_element = assembled.elements[3];
        if (color_element.format == DXGI_FORMAT_B8G8R8A8_UNORM && color_element.buffer < assembled.buffer_count &&
            assembled.buffers[color_element.buffer].src)
            std::memcpy(&draw.color, assembled.buffers[color_element.buffer].src + color_element.offset, 4);
    }
}

bool SaveBmp(const std::wstring& path, uint32_t width, uint32_t height, const uint8_t* pixels, uint32_t pitch) {
    if (!width || !height || !pixels) return false;
    const uint32_t bytes = width * height * 4;
    uint8_t header[54] = {'B', 'M'};
    auto put = [&](int offset, uint32_t value) { std::memcpy(header + offset, &value, sizeof(value)); };
    put(2, 54 + bytes);
    put(10, 54);
    put(14, 40);
    put(18, width);
    put(22, uint32_t(-int32_t(height)));
    header[26] = 1;
    header[28] = 32;
    put(34, bytes);
    FILE* file = _wfopen(path.c_str(), L"wb");
    if (!file) return false;
    const bool written = std::fwrite(header, 1, sizeof(header), file) == sizeof(header);
    for (uint32_t y = 0; written && y < height; ++y)
        if (std::fwrite(pixels + size_t(y) * pitch, 1, width * 4, file) != width * 4) {
            std::fclose(file);
            return false;
        }
    std::fclose(file);
    return written;
}

}  // namespace

VulkanBackend::~VulkanBackend() {
    EndRendering();
    if (context_.Initialized()) {
        auto wait = DeviceProc<PFN_vkDeviceWaitIdle>(context_, "vkDeviceWaitIdle");
        if (!wait) {
            LOGE("vulkan: missing device completion operation during shutdown");
            std::abort();
        }
        const VkResult result = wait(context_.Device());
        if (result != VK_SUCCESS && result != VK_ERROR_DEVICE_LOST) {
            LOGE("vulkan: device completion failed during shutdown (%d)", int(result));
            std::abort();
        }
    }
    for (uint32_t i = 0; i < kFrameContextCount; ++i) {
        active_frame_index_ = i;
        active_frame_ = &frame_contexts_[i];
        if (ActiveFrame().perf_frame_active && ActiveFrame().perf_frame_ended) FinishPerfFrame(0);
    }
    if (perf_enabled_) ReportPerfWindow();
    ResetPerfQueries();
    if (ui_initialized_) {
        std::lock_guard lock(app::UiLock());
        for (const auto& [key, texture] : ui_textures_)
            if (texture.descriptor) ImGui_ImplVulkan_RemoveTexture(texture.descriptor);
        ImGui_ImplVulkan_Shutdown();
        ui_initialized_ = false;
    }
    auto destroy_pool = DeviceProc<PFN_vkDestroyDescriptorPool>(context_, "vkDestroyDescriptorPool");
    if (ui_pool_ && destroy_pool) destroy_pool(context_.Device(), ui_pool_, nullptr);
    ui_pool_ = VK_NULL_HANDLE;
    ui_textures_.clear();
    pending_ui_uploads_.clear();
    pending_screenshot_.clear();
    screenshot_buffer_.reset();
    pending_override_uploads_.clear();
    override_textures_.clear();
    for (auto& frame : frame_contexts_) {
        frame.frame_buffers.clear();
        frame.uploads.Reset();
        frame.frame_images.clear();
        frame.descriptors.Reset();
        frame.batch.Reset();
    }
    // Late readback buffers are also batch-referenced; the batch has retired before their owners are destroyed.
    {
        std::lock_guard lock(late_lock_);
        late_.clear();
    }
    auto destroy_pipeline = DeviceProc<PFN_vkDestroyPipeline>(context_, "vkDestroyPipeline");
    auto destroy_sampler = DeviceProc<PFN_vkDestroySampler>(context_, "vkDestroySampler");
    auto destroy_layout = DeviceProc<PFN_vkDestroyPipelineLayout>(context_, "vkDestroyPipelineLayout");
    if (destroy_pipeline)
        for (const auto& [key, pipeline] : pipelines_)
            if (pipeline) destroy_pipeline(context_.Device(), pipeline, nullptr);
    if (blit_pipeline_ && destroy_pipeline) destroy_pipeline(context_.Device(), blit_pipeline_, nullptr);
    if (destroy_sampler) {
        if (linear_sampler_) destroy_sampler(context_.Device(), linear_sampler_, nullptr);
        if (point_sampler_) destroy_sampler(context_.Device(), point_sampler_, nullptr);
        for (VkSampler sampler : samplers_)
            if (sampler) destroy_sampler(context_.Device(), sampler, nullptr);
    }
    if (pipeline_layout_ && destroy_layout) destroy_layout(context_.Device(), pipeline_layout_, nullptr);
    pipelines_.clear();
    shader_sources_.Clear();
    blit_pipeline_ = VK_NULL_HANDLE;
    linear_sampler_ = point_sampler_ = VK_NULL_HANDLE;
    pipeline_layout_ = VK_NULL_HANDLE;
    sky_scene_color_ = sky_scene_depth_ = nullptr;
    sky_scene_done_ = false;
    textures_.clear();
    targets_.clear();
    dummy_.reset();
    samplers_.clear();
    stage_samplers_.clear();
    custom_border_sampler_count_ = 0;
    swapchain_.Reset();
    surface_.Reset();
    ClearDrawDispatch();
    context_.Reset();
}

bool VulkanBackend::Fail(const char* operation, const std::string& detail) {
    bool expected = false;
    if (terminal_.compare_exchange_strong(expected, true, std::memory_order_acq_rel,
                                          std::memory_order_acquire)) {
        last_error_ = detail;
        LOGE("vulkan: %s failed%s%s", operation, detail.empty() ? "" : ": ", detail.c_str());
    }
    return false;
}

void VulkanBackend::ReportCapabilities() {
    const app::Settings& settings = app::Get();
    uint64_t signature = 1469598103934665603ull;
    auto add = [&](uint64_t value) {
        signature ^= value;
        signature *= 1099511628211ull;
    };
    auto requested = [&](bool enabled, const char* name, std::string& list) {
        add(enabled ? 1 : 0);
        if (enabled) {
            if (!list.empty()) list += ",";
            list += name;
        }
    };
    auto requested_value = [&](uint32_t value, const char* name, std::string& list) {
        add(value);
        if (value) {
            if (!list.empty()) list += ",";
            list += name;
        }
    };

    std::string unsupported;
    // The current Vulkan path has original draw/texture/UI plumbing, but not the D3D12 enhancement pass graph.
    requested(settings.smaa, "smaa", unsupported);
    requested(settings.taa, "taa", unsupported);
    requested(settings.sharpening != 0, "sharpening", unsupported);
    requested(settings.hdr, "hdr", unsupported);
    requested(settings.bloom, "bloom", unsupported);
    requested(settings.ao, "ambient_occlusion", unsupported);
    requested(settings.contact_shadows, "contact_shadows", unsupported);
    requested_value(settings.shadow_quality, "shadow_quality", unsupported);
    requested(settings.dynamic_lights, "dynamic_lights", unsupported);
    requested(settings.light_halos, "light_halos", unsupported);
    requested(settings.soft_particles, "soft_particles", unsupported);
    requested(settings.heat_haze, "heat_haze", unsupported);
    requested(settings.sun_shafts, "sun_shafts", unsupported);
    requested(settings.aim_dof, "aim_depth_of_field", unsupported);
    requested(settings.explosion_flash, "explosion_flash", unsupported);
    requested(settings.flash_shadows, "flash_shadows", unsupported);
    requested(settings.auto_exposure, "auto_exposure", unsupported);
    requested(settings.cloud_shadows, "cloud_shadows", unsupported);
    requested(settings.volumetric_fog, "volumetric_fog", unsupported);
    requested(settings.soft_shadows, "soft_shadows", unsupported);
    requested(settings.bounce_light, "bounce_light", unsupported);
    requested(settings.character_light, "character_light", unsupported);
    requested(settings.lit_smoke, "lit_smoke", unsupported);
    requested(settings.ambient_particles, "ambient_particles", unsupported);
    requested(settings.wind, "wind", unsupported);
    requested(settings.barrel_heat, "barrel_heat", unsupported);
    requested(settings.animated_fire, "animated_fire", unsupported);
    requested(settings.shockwaves, "shockwaves", unsupported);
    requested(settings.water, "water", unsupported);
    requested(settings.water_interaction, "water_interaction", unsupported);
    requested(settings.snow_tracks, "snow_tracks", unsupported);
    requested(settings.lens_flare, "lens_flare", unsupported);
    requested(settings.weapon_materials, "weapon_materials", unsupported);
    requested(settings.character_detail, "character_detail", unsupported);
    requested(settings.color_grade, "color_grade", unsupported);
    requested(settings.film_grain, "film_grain", unsupported);
    requested(settings.reflections, "reflections", unsupported);
    requested(settings.smooth_models, "smooth_models", unsupported);
    requested(settings.sun_moon, "sun_moon", unsupported);
    requested(settings.clouds, "clouds", unsupported);
    requested(settings.materials, "materials", unsupported);
    requested(settings.parallax, "parallax", unsupported);
    requested(settings.detail, "detail", unsupported);
    requested(settings.pixel_lighting, "pixel_lighting", unsupported);
    requested(settings.height_blend, "height_blend", unsupported);
    requested(settings.sharp_distance, "sharp_distance", unsupported);
    add(settings.aniso);
    if (settings.aniso) {
        if (!unsupported.empty()) unsupported += ",";
        unsupported += "anisotropic_filtering";
    }
    requested(settings.remaster, "remaster_material_passes", unsupported);

    if (capability_reported_ && capability_settings_signature_ == signature) return;
    capability_reported_ = true;
    capability_settings_signature_ = signature;
    const uint32_t api = context_.Properties().apiVersion;
    LOGR("vulkan: capability report tier=original-core original_control=available remaster_pass_graph=unavailable settings_signature=%016llX api=%u.%u.%u validation=%s mirror_clamp=%s swapchain_maintenance1=%s custom_border_color_extension=%s custom_border_color_feature=%s custom_border_color=%s custom_border_color_without_format=%s max_custom_border_color_samplers=%u",
         static_cast<unsigned long long>(signature), VK_VERSION_MAJOR(api), VK_VERSION_MINOR(api), VK_VERSION_PATCH(api),
         context_.ValidationEnabled() ? "enabled" : context_.ValidationAvailable() ? "available-not-enabled" : "unavailable",
         context_.SamplerMirrorClampToEdgeEnabled() ? "enabled" : "unavailable",
         context_.SwapchainMaintenance1Enabled() ? "enabled" : "unavailable",
         context_.CustomBorderColorExtensionAvailable() ? "available" : "unavailable",
         context_.CustomBorderColorFeatureAvailable() ? "available" : "unavailable",
         context_.CustomBorderColorEnabled() ? "enabled" : "unavailable",
         context_.CustomBorderColorWithoutFormatEnabled() ? "enabled" : "unavailable",
         context_.MaxCustomBorderColorSamplers());
    if (!unsupported.empty())
        LOGW("vulkan: requested settings are not executed by the current Vulkan enhancement pass graph (values preserved): %s",
             unsupported.c_str());
}

bool VulkanBackend::InitPerfQueries(uint32_t forced_interval) {
    const char* value = std::getenv("XBE_VK_PERF");
    const bool trace_requested = diag::PerfTrace::Requested();
    const char* detail = std::getenv("XBE_VK_PERF_DETAIL");
    perf_detail_enabled_ = trace_requested && detail && std::strcmp(detail, "1") == 0;
    if (!forced_interval && (!value || !value[0] || value[0] == '0') && !trace_requested) return true;
    perf_enabled_ = true;
    perf_counters_enabled_.store(true, std::memory_order_relaxed);
    perf_interval_ = forced_interval ? forced_interval :
                                      value && value[0] && value[0] != '0'
                                          ? std::max(1u, static_cast<uint32_t>(std::strtoul(value, nullptr, 10)))
                                          : 600;
    if (!context_.Properties().limits.timestampPeriod) {
        LOGW("vulkan: XBE_VK_PERF enabled but timestamp queries are unavailable");
        return true;
    }
    auto create = DeviceProc<PFN_vkCreateQueryPool>(context_, "vkCreateQueryPool");
    if (!create) {
        LOGW("vulkan: XBE_VK_PERF enabled but vkCreateQueryPool is unavailable");
        return true;
    }
    VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    info.queryCount = 2;
    std::string error;
    bool created = true;
    for (auto& frame : frame_contexts_) {
        if (!Result(create(context_.Device(), &info, nullptr, &frame.perf_query_pool), "vkCreateQueryPool", error)) {
            created = false;
            break;
        }
    }
    if (!created) {
        LOGW("vulkan: XBE_VK_PERF GPU timestamps disabled: %s", error.c_str());
        ResetPerfQueries();
    } else {
        LOGR("vulkan: perf diagnostics enabled interval=%u gpu_timestamps=on detail=%s", perf_interval_,
             perf_detail_enabled_ ? "trace-opt-in" : "off");
    }
    return true;
}

bool VulkanBackend::LoadDrawDispatch() {
    std::string error;
    if (!draw_dispatch_.Load([this](const char* name) { return context_.DeviceProc(name); }, error))
        return Fail("draw dispatch initialization", error);
    return true;
}

void VulkanBackend::ClearDrawDispatch() {
    draw_dispatch_.Clear();
}

void VulkanBackend::ResetPerfQueries() {
    auto destroy = DeviceProc<PFN_vkDestroyQueryPool>(context_, "vkDestroyQueryPool");
    for (auto& frame : frame_contexts_) {
        if (frame.perf_query_pool && destroy) destroy(context_.Device(), frame.perf_query_pool, nullptr);
        frame.perf_query_pool = VK_NULL_HANDLE;
        frame.perf_frame_active = false;
        frame.perf_detail_frame_active = false;
        frame.perf_frame_ended = false;
        frame.perf_query_pending = false;
    }
}

void VulkanBackend::BeginPerfFrame() {
    if (!perf_enabled_) return;
    ActiveFrame().perf_current_draws = ActiveFrame().perf_current_pipeline_misses = ActiveFrame().perf_current_shader_modules = 0;
    ActiveFrame().perf_current_source_hits = ActiveFrame().perf_current_source_misses = 0;
    ActiveFrame().perf_current_texture_uploads = ActiveFrame().perf_current_descriptor_requests = 0;
    ActiveFrame().perf_current_record_ns = ActiveFrame().perf_current_wait_ns = 0;
    ActiveFrame().perf_current_detail_vertex_assembly_ns = 0;
    ActiveFrame().perf_current_detail_texture_target_sampler_ns = 0;
    ActiveFrame().perf_current_detail_pipeline_source_ns = 0;
    ActiveFrame().perf_current_detail_constant_vertex_index_upload_ns = 0;
    ActiveFrame().perf_current_detail_descriptor_lookup_allocation_ns = 0;
    ActiveFrame().perf_current_detail_command_recording_ns = 0;
    ActiveFrame().perf_current_detail_swapchain_render_wait_ns = 0;
    ActiveFrame().perf_current_detail_batch_completion_wait_ns = 0;
    ActiveFrame().perf_current_detail_late_publish_ns = 0;
    ActiveFrame().perf_current_detail_descriptor_retire_ns = 0;
    ActiveFrame().perf_current_detail_resource_retire_ns = 0;
    ActiveFrame().perf_current_upload_allocations = ActiveFrame().perf_current_upload_bytes = ActiveFrame().perf_current_upload_block_grows = 0;
    ActiveFrame().perf_current_gpu_valid = false;
    ActiveFrame().perf_current_gpu_ms = 0.0;
    ActiveFrame().perf_frame_start = std::chrono::steady_clock::now();
    ActiveFrame().perf_frame_active = true;
    ActiveFrame().perf_detail_frame_active = perf_detail_enabled_ && g_frame_stats.load(std::memory_order_relaxed) == 1;
    ActiveFrame().perf_frame_ended = false;
    ActiveFrame().perf_query_pending = false;
    if (!ActiveFrame().perf_query_pool) return;
    auto reset = DeviceProc<PFN_vkCmdResetQueryPool>(context_, "vkCmdResetQueryPool");
    auto write = DeviceProc<PFN_vkCmdWriteTimestamp>(context_, "vkCmdWriteTimestamp");
    if (!reset || !write) {
        LOGW("vulkan: XBE_VK_PERF GPU timestamps disabled after missing command entry point");
        ResetPerfQueries();
        return;
    }
    reset(ActiveFrame().command, ActiveFrame().perf_query_pool, 0, 2);
    write(ActiveFrame().command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, ActiveFrame().perf_query_pool, 0);
    ActiveFrame().perf_query_pending = true;
}

void VulkanBackend::EndPerfFrame() {
    if (!perf_enabled_ || !ActiveFrame().perf_frame_active || ActiveFrame().perf_frame_ended) return;
    ActiveFrame().perf_current_record_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - ActiveFrame().perf_frame_start).count());
    if (ActiveFrame().perf_query_pending) {
        auto write = DeviceProc<PFN_vkCmdWriteTimestamp>(context_, "vkCmdWriteTimestamp");
        if (write) write(ActiveFrame().command, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ActiveFrame().perf_query_pool, 1);
    }
    ActiveFrame().perf_frame_ended = true;
}

void VulkanBackend::ReportPerfWindow() {
    if (!perf_window_frames_) return;
    const uint64_t sync_requests = perf_window_sync_requests_.exchange(0, std::memory_order_relaxed);
    const uint64_t deliver_late_hits = perf_window_deliver_late_hits_.exchange(0, std::memory_order_relaxed);
    const uint64_t deliver_late_misses = perf_window_deliver_late_misses_.exchange(0, std::memory_order_relaxed);
    const uint64_t record_late_accepted = perf_window_record_late_accepted_.exchange(0, std::memory_order_relaxed);
    const uint64_t record_late_skips = perf_window_record_late_skips_.exchange(0, std::memory_order_relaxed);
    const uint64_t guest_writeback_successes =
        perf_window_guest_writeback_successes_.exchange(0, std::memory_order_relaxed);
    const vulkan::ShaderCacheStats shader_cache_stats = shader_cache_.Stats();
    const vulkan::PipelineCacheStats pipeline_cache_stats = context_.PipelineCacheStatus();
    const double frames = double(perf_window_frames_);
    const auto ms = [](uint64_t ns) { return double(ns) / 1000000.0; };
    if (perf_window_gpu_samples_) {
        LOGR("vulkan: perf frames=%llu draws=%llu pipelines=%llu shaders=%llu textures=%llu descriptor_requests=%llu "
             "sources_hit=%llu sources_miss=%llu upload_allocs=%llu upload_kib=%.1f block_grows=%llu rewinds=%llu peak_blocks=%zu "
             "record_ms=%.2f wait_ms=%.2f gpu_ms=%.2f sync_requests=%llu late_deliver_hits=%llu "
             "late_deliver_misses=%llu late_record_accepted=%llu late_record_skips=%llu guest_writeback_successes=%llu "
             "shader_cache_hits=%llu shader_cache_misses=%llu shader_cache_compiler_invocations=%llu "
             "pipeline_cache_enabled=%s pipeline_cache_loaded=%s",
             (unsigned long long)perf_window_frames_, (unsigned long long)perf_window_draws_,
             (unsigned long long)perf_window_pipeline_misses_, (unsigned long long)perf_window_shader_modules_,
             (unsigned long long)perf_window_texture_uploads_, (unsigned long long)perf_window_descriptor_requests_,
             (unsigned long long)perf_window_source_hits_, (unsigned long long)perf_window_source_misses_,
             (unsigned long long)perf_window_upload_allocations_, double(perf_window_upload_bytes_) / 1024.0,
             (unsigned long long)perf_window_upload_block_grows_, (unsigned long long)perf_window_upload_rewinds_,
             perf_window_peak_upload_blocks_,
             ms(perf_window_record_ns_) / frames, ms(perf_window_wait_ns_) / frames,
             perf_window_gpu_ms_ / double(perf_window_gpu_samples_), (unsigned long long)sync_requests,
             (unsigned long long)deliver_late_hits, (unsigned long long)deliver_late_misses,
             (unsigned long long)record_late_accepted, (unsigned long long)record_late_skips,
             (unsigned long long)guest_writeback_successes,
             (unsigned long long)shader_cache_stats.hits, (unsigned long long)shader_cache_stats.misses,
             (unsigned long long)shader_cache_stats.compiler_invocations,
             pipeline_cache_stats.enabled ? "1" : "0", pipeline_cache_stats.loaded_from_disk ? "1" : "0");
    } else {
        LOGR("vulkan: perf frames=%llu draws=%llu pipelines=%llu shaders=%llu textures=%llu descriptor_requests=%llu "
             "sources_hit=%llu sources_miss=%llu upload_allocs=%llu upload_kib=%.1f block_grows=%llu rewinds=%llu peak_blocks=%zu "
             "record_ms=%.2f wait_ms=%.2f gpu_ms=unavailable sync_requests=%llu late_deliver_hits=%llu "
             "late_deliver_misses=%llu late_record_accepted=%llu late_record_skips=%llu guest_writeback_successes=%llu "
             "shader_cache_hits=%llu shader_cache_misses=%llu shader_cache_compiler_invocations=%llu "
             "pipeline_cache_enabled=%s pipeline_cache_loaded=%s",
             (unsigned long long)perf_window_frames_, (unsigned long long)perf_window_draws_,
             (unsigned long long)perf_window_pipeline_misses_, (unsigned long long)perf_window_shader_modules_,
             (unsigned long long)perf_window_texture_uploads_, (unsigned long long)perf_window_descriptor_requests_,
             (unsigned long long)perf_window_source_hits_, (unsigned long long)perf_window_source_misses_,
             (unsigned long long)perf_window_upload_allocations_, double(perf_window_upload_bytes_) / 1024.0,
             (unsigned long long)perf_window_upload_block_grows_, (unsigned long long)perf_window_upload_rewinds_,
             perf_window_peak_upload_blocks_, ms(perf_window_record_ns_) / frames, ms(perf_window_wait_ns_) / frames,
             (unsigned long long)sync_requests, (unsigned long long)deliver_late_hits,
             (unsigned long long)deliver_late_misses, (unsigned long long)record_late_accepted,
             (unsigned long long)record_late_skips, (unsigned long long)guest_writeback_successes,
             (unsigned long long)shader_cache_stats.hits, (unsigned long long)shader_cache_stats.misses,
             (unsigned long long)shader_cache_stats.compiler_invocations,
             pipeline_cache_stats.enabled ? "1" : "0", pipeline_cache_stats.loaded_from_disk ? "1" : "0");
    }
    perf_window_frames_ = perf_window_draws_ = perf_window_pipeline_misses_ = perf_window_shader_modules_ = 0;
    perf_window_source_hits_ = perf_window_source_misses_ = 0;
    perf_window_texture_uploads_ = perf_window_descriptor_requests_ = 0;
    perf_window_record_ns_ = perf_window_wait_ns_ = 0;
    perf_window_upload_allocations_ = perf_window_upload_bytes_ = perf_window_upload_block_grows_ = 0;
    perf_window_upload_rewinds_ = 0;
    perf_window_peak_upload_blocks_ = 0;
    perf_window_gpu_samples_ = 0;
    perf_window_gpu_ms_ = 0.0;
}

void VulkanBackend::FinishPerfFrame(uint64_t wait_ns) {
    if (!perf_enabled_ || !ActiveFrame().perf_frame_active || !ActiveFrame().perf_frame_ended) return;
    if (g_frame_stats.load() == 2) perf_trace_.Stop();
    ActiveFrame().perf_current_wait_ns = wait_ns;
    ActiveFrame().perf_current_frame_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ActiveFrame().perf_frame_start).count();
    if (ActiveFrame().perf_query_pending) {
        uint64_t timestamps[2] = {};
        auto get = DeviceProc<PFN_vkGetQueryPoolResults>(context_, "vkGetQueryPoolResults");
        const VkResult result = get ? get(context_.Device(), ActiveFrame().perf_query_pool, 0, 2, sizeof(timestamps), timestamps,
                                          sizeof(uint64_t), VK_QUERY_RESULT_64_BIT)
                                    : VK_ERROR_INITIALIZATION_FAILED;
        if (result == VK_SUCCESS && timestamps[1] >= timestamps[0]) {
            ActiveFrame().perf_current_gpu_valid = true;
            ActiveFrame().perf_current_gpu_ms = double(timestamps[1] - timestamps[0]) *
                                   double(context_.Properties().limits.timestampPeriod) / 1000000.0;
        }
    }
    ++perf_window_frames_;
    perf_window_draws_ += ActiveFrame().perf_current_draws;
    perf_window_pipeline_misses_ += ActiveFrame().perf_current_pipeline_misses;
    perf_window_shader_modules_ += ActiveFrame().perf_current_shader_modules;
    perf_window_source_hits_ += ActiveFrame().perf_current_source_hits;
    perf_window_source_misses_ += ActiveFrame().perf_current_source_misses;
    perf_window_texture_uploads_ += ActiveFrame().perf_current_texture_uploads;
    perf_window_descriptor_requests_ += ActiveFrame().perf_current_descriptor_requests;
    perf_window_record_ns_ += ActiveFrame().perf_current_record_ns;
    perf_window_wait_ns_ += ActiveFrame().perf_current_wait_ns;
    perf_window_upload_allocations_ += ActiveFrame().perf_current_upload_allocations;
    perf_window_upload_bytes_ += ActiveFrame().perf_current_upload_bytes;
    perf_window_upload_block_grows_ += ActiveFrame().perf_current_upload_block_grows;
    perf_window_peak_upload_blocks_ = std::max(perf_window_peak_upload_blocks_, ActiveFrame().uploads.BlockCount());
    if (ActiveFrame().perf_current_gpu_valid) {
        ++perf_window_gpu_samples_;
        perf_window_gpu_ms_ += ActiveFrame().perf_current_gpu_ms;
    }
    if (g_frame_stats.load() == 1 && ActiveFrame().perf_current_frame_ms > 0.0 && perf_trace_.Enabled()) {
        diag::PerfFrameSample sample;
        sample.index = perf_trace_frame_index_++;
        sample.renderer_sample_ms = ActiveFrame().perf_current_frame_ms;
        sample.record_ms = double(ActiveFrame().perf_current_record_ns) / 1000000.0;
        sample.wait_ms = double(ActiveFrame().perf_current_wait_ns) / 1000000.0;
        if (ActiveFrame().perf_current_gpu_valid) sample.gpu_ms = ActiveFrame().perf_current_gpu_ms;
        sample.draws = ActiveFrame().perf_current_draws;
        sample.shader_events = ActiveFrame().perf_current_shader_modules;
        sample.pipeline_events = ActiveFrame().perf_current_pipeline_misses;
        sample.texture_uploads = ActiveFrame().perf_current_texture_uploads;
        sample.auxiliary_events = ActiveFrame().perf_current_descriptor_requests;
        sample.source_hits = ActiveFrame().perf_current_source_hits;
        sample.source_misses = ActiveFrame().perf_current_source_misses;
        if (ActiveFrame().perf_detail_frame_active) {
            const auto detail_ms = [](uint64_t ns) { return double(ns) / 1000000.0; };
            const uint64_t phase_sum_ns = ActiveFrame().perf_current_detail_vertex_assembly_ns +
                                          ActiveFrame().perf_current_detail_texture_target_sampler_ns +
                                          ActiveFrame().perf_current_detail_pipeline_source_ns +
                                          ActiveFrame().perf_current_detail_constant_vertex_index_upload_ns +
                                          ActiveFrame().perf_current_detail_descriptor_lookup_allocation_ns +
                                          ActiveFrame().perf_current_detail_command_recording_ns;
            sample.detail_vertex_assembly_ms = detail_ms(ActiveFrame().perf_current_detail_vertex_assembly_ns);
            sample.detail_texture_target_sampler_ms = detail_ms(ActiveFrame().perf_current_detail_texture_target_sampler_ns);
            sample.detail_pipeline_source_ms = detail_ms(ActiveFrame().perf_current_detail_pipeline_source_ns);
            sample.detail_constant_vertex_index_upload_ms =
                detail_ms(ActiveFrame().perf_current_detail_constant_vertex_index_upload_ns);
            sample.detail_descriptor_lookup_allocation_ms =
                detail_ms(ActiveFrame().perf_current_detail_descriptor_lookup_allocation_ns);
            sample.detail_command_recording_ms = detail_ms(ActiveFrame().perf_current_detail_command_recording_ns);
            sample.detail_phase_sum_ms = detail_ms(phase_sum_ns);
            sample.detail_record_residual_ms = (double(ActiveFrame().perf_current_record_ns) - double(phase_sum_ns)) / 1000000.0;
            const uint64_t wait_phase_sum_ns = ActiveFrame().perf_current_detail_swapchain_render_wait_ns +
                                               ActiveFrame().perf_current_detail_batch_completion_wait_ns +
                                               ActiveFrame().perf_current_detail_late_publish_ns +
                                               ActiveFrame().perf_current_detail_descriptor_retire_ns +
                                               ActiveFrame().perf_current_detail_resource_retire_ns;
            sample.detail_swapchain_render_wait_ms = detail_ms(ActiveFrame().perf_current_detail_swapchain_render_wait_ns);
            sample.detail_batch_completion_wait_ms = detail_ms(ActiveFrame().perf_current_detail_batch_completion_wait_ns);
            sample.detail_late_publish_ms = detail_ms(ActiveFrame().perf_current_detail_late_publish_ns);
            sample.detail_descriptor_retire_ms = detail_ms(ActiveFrame().perf_current_detail_descriptor_retire_ns);
            sample.detail_resource_retire_ms = detail_ms(ActiveFrame().perf_current_detail_resource_retire_ns);
            sample.detail_wait_phase_sum_ms = detail_ms(wait_phase_sum_ns);
            sample.detail_wait_residual_ms =
                (double(ActiveFrame().perf_current_wait_ns) - double(wait_phase_sum_ns)) / 1000000.0;
        }
        perf_trace_.Frame(sample);
    }
    if (perf_window_frames_ >= perf_interval_) ReportPerfWindow();
    ActiveFrame().perf_frame_active = false;
    ActiveFrame().perf_detail_frame_active = false;
    ActiveFrame().perf_frame_ended = false;
    ActiveFrame().perf_query_pending = false;
}

bool VulkanBackend::Init(uint32_t width, uint32_t height) {
    guest_width_ = width;
    guest_height_ = height;
    render_scale_ = std::max(app::RenderScale(height), 1.0f);
    std::string error;
    const std::filesystem::path cache_directory = app::ExeDir() / "shadercache" / "vulkan";
    context_.SetPipelineCacheDirectory(cache_directory);
    if (!context_.Init(error, true)) return Fail("context initialization", error);
    shader_cache_.SetDirectory(cache_directory);
    if (!LoadDrawDispatch()) return false;
    for (auto& frame : frame_contexts_)
        if (!frame.uploads.Init(context_, 4u * 1024u * 1024u, error))
            return Fail("upload allocator initialization", error);
    if (!surface_.Init(context_, static_cast<HWND>(app::Window()), error)) return Fail("Win32 surface initialization", error);
    if (!surface_.SupportsGraphicsQueue(context_, error)) return Fail("Win32 queue support", error);
    app::Mode output = app::Output();
    VkExtent2D extent{std::max(output.width, 1u), std::max(output.height, 1u)};
    if (swapchain_.Init(context_, surface_, extent, app::Get().vsync ? vulkan::PresentPreference::Fifo
                                                                       : vulkan::PresentPreference::Mailbox,
                       error) != vulkan::SwapchainInitStatus::Ready)
        return Fail("swapchain initialization", error);
    for (auto& frame : frame_contexts_)
        if (!frame.descriptors.Init(context_, 256, error)) return Fail("descriptor ABI initialization", error);
    for (auto& frame : frame_contexts_)
        if (!frame.batch.Init(context_, error)) return Fail("command batch initialization", error);
    if (!InitPerfQueries()) return false;
    if (!CreatePipelineLayout()) return false;
    if (!CreateSamplers()) return false;
    if (!BeginFrame()) return false;
    if (!CreateDummy()) return false;
    if (!CreateBlitPipeline()) return false;
    if (!CreateUi()) return false;
    ReportCapabilities();
    LOGR("vulkan: core backend initialized for %ux%u guest at %.3fx render scale (%ux%u output)", width, height,
         render_scale_, swapchain_.Extent().width, swapchain_.Extent().height);
    return true;
}

bool VulkanBackend::BeginFrame() {
    if (terminal_) return false;
    ReportCapabilities();
    PublishLateCompletion();
    if (terminal_) return false;
    if (ActiveFrame().recording) return true;
    if (frame_started_) {
        active_frame_index_ = (active_frame_index_ + 1) % kFrameContextCount;
        active_frame_ = &frame_contexts_[active_frame_index_];
    } else {
        frame_started_ = true;
    }
    if (!perf_enabled_ && g_log_draws.load(std::memory_order_relaxed) >= 3) InitPerfQueries(3);
    const auto wait_start = std::chrono::steady_clock::now();
    const bool capture_wait_detail = ActiveFrame().perf_frame_active && ActiveFrame().perf_detail_frame_active;
    std::string error;
    if (ActiveFrame().batch.Pending()) {
        const auto batch_wait_start = capture_wait_detail ? std::chrono::steady_clock::now()
                                                          : std::chrono::steady_clock::time_point{};
        if (!ActiveFrame().batch.Wait(UINT64_MAX, error)) return Fail("frame completion", error);
        if (capture_wait_detail) ActiveFrame().perf_current_detail_batch_completion_wait_ns += ElapsedNanoseconds(batch_wait_start);
        completed_submission_ordinal_ = std::max(completed_submission_ordinal_, ActiveFrame().submission_ordinal);
        const auto late_publish_start = capture_wait_detail ? std::chrono::steady_clock::now()
                                                            : std::chrono::steady_clock::time_point{};
        PublishLateCompletion();
        if (capture_wait_detail) ActiveFrame().perf_current_detail_late_publish_ns += ElapsedNanoseconds(late_publish_start);
        if (ActiveFrame().descriptors_used) {
            const auto descriptor_retire_start = capture_wait_detail ? std::chrono::steady_clock::now()
                                                                     : std::chrono::steady_clock::time_point{};
            if (!ActiveFrame().descriptors.ResetAfterCompletion(ActiveFrame().batch, 0, error)) return Fail("descriptor retirement", error);
            if (capture_wait_detail)
                ActiveFrame().perf_current_detail_descriptor_retire_ns += ElapsedNanoseconds(descriptor_retire_start);
        }
        ActiveFrame().descriptors_used = false;
        const auto resource_retire_start = capture_wait_detail ? std::chrono::steady_clock::now()
                                                               : std::chrono::steady_clock::time_point{};
        ActiveFrame().frame_buffers.clear();
        RetireUploads();
        ActiveFrame().frame_images.clear();
        if (capture_wait_detail) ActiveFrame().perf_current_detail_resource_retire_ns += ElapsedNanoseconds(resource_retire_start);
    }
    const uint64_t wait_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - wait_start).count());
    if (ActiveFrame().perf_frame_active && ActiveFrame().perf_frame_ended) FinishPerfFrame(wait_ns);
    ActiveFrame().submission_ordinal = 0;
    ActiveFrame().completion_fence = VK_NULL_HANDLE;
    if (!ActiveFrame().batch.Begin(ActiveFrame().command, error)) return Fail("command recording begin", error);
    ++frame_generation_;
    ActiveFrame().recording = true;
    BeginPerfFrame();
    if (!PumpOverrides()) return false;
    return true;
}

bool VulkanBackend::CompleteFrame() {
    PublishLateCompletion();
    if (terminal_) return false;
    EndRendering();
    EndPerfFrame();
    std::string error;
    if (ActiveFrame().recording) {
        if (!ActiveFrame().batch.Submit(error)) {
            if (!terminal_) return Fail("ordinary command submit", error);
            ActiveFrame().batch.Reset();
            ActiveFrame().recording = false;
            return false;
        }
        ActiveFrame().submission_ordinal = ++next_submission_ordinal_;
        StampLateSubmissions();
        ActiveFrame().recording = false;
        if (ActiveFrame().descriptors_used && !ActiveFrame().descriptors.MarkInFlight(ActiveFrame().batch, error)) return Fail("ordinary descriptor ownership", error);
    }
    return true;
}

bool VulkanBackend::RetireFrameContext(uint32_t index, uint64_t timeout_ns, const char* operation) {
    if (index >= kFrameContextCount) return false;
    const uint32_t saved_index = active_frame_index_;
    active_frame_index_ = index;
    active_frame_ = &frame_contexts_[index];
    if (ActiveFrame().recording) {
        active_frame_index_ = saved_index;
        active_frame_ = &frame_contexts_[saved_index];
        return Fail(operation, "attempted to retire a frame context that is still recording");
    }
    const auto wait_start = std::chrono::steady_clock::now();
    const bool capture_wait_detail = ActiveFrame().perf_frame_active && ActiveFrame().perf_detail_frame_active;
    std::string error;
    if (ActiveFrame().batch.Pending()) {
        const auto batch_wait_start = capture_wait_detail ? std::chrono::steady_clock::now()
                                                          : std::chrono::steady_clock::time_point{};
        if (!ActiveFrame().batch.Wait(timeout_ns, error)) {
            active_frame_index_ = saved_index;
            active_frame_ = &frame_contexts_[saved_index];
            return Fail(operation, error);
        }
        if (capture_wait_detail)
            ActiveFrame().perf_current_detail_batch_completion_wait_ns += ElapsedNanoseconds(batch_wait_start);
        completed_submission_ordinal_ = std::max(completed_submission_ordinal_, ActiveFrame().submission_ordinal);
        PublishLateCompletion();
    }
    if (ActiveFrame().descriptors_used) {
        const auto descriptor_retire_start = capture_wait_detail ? std::chrono::steady_clock::now()
                                                                 : std::chrono::steady_clock::time_point{};
        if (!ActiveFrame().descriptors.ResetAfterCompletion(ActiveFrame().batch, 0, error)) {
            active_frame_index_ = saved_index;
            active_frame_ = &frame_contexts_[saved_index];
            return Fail(operation, error);
        }
        if (capture_wait_detail)
            ActiveFrame().perf_current_detail_descriptor_retire_ns += ElapsedNanoseconds(descriptor_retire_start);
    }
    ActiveFrame().descriptors_used = false;
    const auto resource_retire_start = capture_wait_detail ? std::chrono::steady_clock::now()
                                                           : std::chrono::steady_clock::time_point{};
    ActiveFrame().frame_buffers.clear();
    RetireUploads();
    ActiveFrame().frame_images.clear();
    if (capture_wait_detail)
        ActiveFrame().perf_current_detail_resource_retire_ns += ElapsedNanoseconds(resource_retire_start);
    const uint64_t wait_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - wait_start).count());
    if (ActiveFrame().perf_frame_active && ActiveFrame().perf_frame_ended) FinishPerfFrame(wait_ns);
    ActiveFrame().submission_ordinal = 0;
    ActiveFrame().completion_fence = VK_NULL_HANDLE;
    active_frame_index_ = saved_index;
    active_frame_ = &frame_contexts_[saved_index];
    return !terminal_;
}

bool VulkanBackend::DrainFrameContexts() {
    EndRendering();
    EndPerfFrame();
    std::string error;
    if (ActiveFrame().recording) {
        if (!ActiveFrame().batch.Submit(error)) return Fail("drain command submit", error);
        ActiveFrame().submission_ordinal = ++next_submission_ordinal_;
        StampLateSubmissions();
        ActiveFrame().recording = false;
        if (ActiveFrame().descriptors_used && !ActiveFrame().descriptors.MarkInFlight(ActiveFrame().batch, error))
            return Fail("drain descriptor ownership", error);
    }
    for (uint32_t i = 0; i < kFrameContextCount; ++i)
        if (!RetireFrameContext(i, UINT64_MAX, "drain frame completion")) return false;
    if (swapchain_.Ready() && !swapchain_.WaitForRendering(UINT64_MAX, error))
        return Fail("drain presentation completion", error);
    PublishLateCompletion();
    return !terminal_;
}

void VulkanBackend::SetGamma(const uint8_t* ramp) {
    if (ramp) {
        if (std::memcmp(gamma_, ramp, sizeof(gamma_)) != 0) {
            std::memcpy(gamma_, ramp, sizeof(gamma_));
            gamma_dirty_ = true;
        }
    }
}

void VulkanBackend::UpdateGammaRamp() {
    const uint32_t brightness = app::Get().brightness;
    if (!gamma_dirty_ && brightness_ == brightness) return;
    brightness_ = brightness;
    gamma_dirty_ = false;
    BuildGammaLut(gamma_, brightness_, gamma_lut_, gamma_effective_);
}

bool VulkanBackend::CreatePipelineLayout() {
    auto create = DeviceProc<PFN_vkCreatePipelineLayout>(context_, "vkCreatePipelineLayout");
    if (!create) return Fail("pipeline layout entry point");
    VkPipelineLayoutCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    VkDescriptorSetLayout set = ActiveFrame().descriptors.Layout();
    info.setLayoutCount = 1;
    info.pSetLayouts = &set;
    std::string error;
    if (!Result(create(context_.Device(), &info, nullptr, &pipeline_layout_), "vkCreatePipelineLayout", error))
        return Fail("pipeline layout", error);
    return true;
}

bool VulkanBackend::CreateSamplers() {
    auto create = DeviceProc<PFN_vkCreateSampler>(context_, "vkCreateSampler");
    if (!create) return Fail("sampler entry point");
    VkSamplerCreateInfo info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    info.magFilter = info.minFilter = VK_FILTER_LINEAR;
    info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    info.addressModeU = info.addressModeV = info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    info.maxLod = VK_LOD_CLAMP_NONE;
    std::string error;
    if (!Result(create(context_.Device(), &info, nullptr, &linear_sampler_), "vkCreateSampler", error))
        return Fail("linear sampler", error);
    info.magFilter = info.minFilter = VK_FILTER_NEAREST;
    info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    if (!Result(create(context_.Device(), &info, nullptr, &point_sampler_), "vkCreateSampler", error))
        return Fail("point sampler", error);
    return true;
}

bool VulkanBackend::CreateDummy() {
    dummy_ = std::make_unique<vulkan::Image>();
    std::string error;
    if (!dummy_->Init(context_, 1, 1, kColorFormat, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                      ColorAspect(), VK_SAMPLE_COUNT_1_BIT, error))
        return Fail("dummy image", error);
    TextureDesc desc{};
    desc.data = 0;
    desc.width = desc.height = 1;
    desc.format = 0x12;
    std::array<uint8_t, 4> black = {0, 0, 0, 255};
    auto buffer = std::make_unique<vulkan::Buffer>();
    if (!buffer->Init(context_, black.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, error) ||
        !buffer->Write(0, black.data(), black.size(), error))
        return Fail("dummy upload buffer", error);
    VkCommandBuffer command = ActiveFrame().command;
    Transition(*dummy_, dummy_layout_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, ColorAspect());
    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = {1, 1, 1};
    auto copy_image = DeviceProc<PFN_vkCmdCopyBufferToImage>(context_, "vkCmdCopyBufferToImage");
    if (!copy_image) return Fail("dummy upload command entry point");
    copy_image(command, buffer->Handle(), dummy_->Handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    Transition(*dummy_, dummy_layout_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, ColorAspect());
    ActiveFrame().frame_buffers.push_back(std::move(buffer));
    return true;
}

bool VulkanBackend::UploadGuestImage(vulkan::Image& image, VkImageLayout& layout, const TextureDesc& desc,
                                     const uint32_t* palette) {
    const FormatInfo& format = Format(desc.format);
    if (format.host == Host::None || !desc.width || !desc.height || !desc.depth || !desc.levels ||
        (desc.cube && desc.depth != 1))
        return Fail("texture upload", "unsupported texture shape or format");
    std::vector<uint8_t> decoded;
    std::vector<VkBufferImageCopy> copies;
    const uint8_t* guest = static_cast<const uint8_t*>(kernel::HostPtr(desc.data));
    const uint32_t faces = desc.cube ? 6 : 1;
    for (uint32_t level = 0; level < desc.levels; ++level) {
        const uint32_t width = std::max(desc.width >> level, 1u);
        const uint32_t height = std::max(desc.height >> level, 1u);
        const uint32_t depth = desc.linear ? 1 : std::max(desc.depth >> level, 1u);
        for (uint32_t face = 0; face < faces; ++face) {
            std::vector<uint8_t> level_data;
            uint32_t pitch = 0;
            DecodeLevel(desc, level, face, guest, palette, level_data, pitch);
            if (level_data.empty() || (level_data.size() & 3))
                return Fail("texture upload", "decoded guest texture level is empty or misaligned");
            VkBufferImageCopy copy{};
            copy.bufferOffset = decoded.size();
            copy.imageSubresource = {static_cast<VkImageAspectFlags>(format.depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT),
                                     level, face, 1};
            copy.imageExtent = {width, height, depth};
            copies.push_back(copy);
            decoded.insert(decoded.end(), level_data.begin(), level_data.end());
        }
    }
    if (decoded.empty()) return Fail("texture upload", "decoded guest texture is empty");
    std::string error;
    auto buffer = std::make_unique<vulkan::Buffer>();
    if (!buffer->Init(context_, decoded.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, error) ||
        !buffer->Write(0, decoded.data(), decoded.size(), error))
        return Fail("texture staging buffer", error);
    const VkImageAspectFlags aspect = format.depth ? VK_IMAGE_ASPECT_DEPTH_BIT : ColorAspect();
    EndRendering();
    Transition(image, layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, aspect);
    auto copy_image = DeviceProc<PFN_vkCmdCopyBufferToImage>(context_, "vkCmdCopyBufferToImage");
    if (!copy_image) return Fail("texture upload command entry point");
    copy_image(ActiveFrame().command, buffer->Handle(), image.Handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
               static_cast<uint32_t>(copies.size()), copies.data());
    Transition(image, layout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, aspect);
    ActiveFrame().frame_buffers.push_back(std::move(buffer));
    if (perf_enabled_) ++ActiveFrame().perf_current_texture_uploads;
    return true;
}

bool VulkanBackend::UploadSurface(Target& target) {
    if (target.depth || !target.surface.data) return true;
    TextureDesc desc;
    if (!DescribeSurface(target.surface, desc)) return Fail("surface upload", "render surface description is invalid");
    const FormatInfo& format = Format(desc.format);
    if (format.host != Host::BGRA8 || desc.cube || desc.depth != 1 || desc.levels == 0)
        return Fail("surface upload", "render surface format cannot be uploaded to the Vulkan color target");

    const uint8_t* guest = static_cast<const uint8_t*>(kernel::HostPtr(desc.data));
    std::vector<uint8_t> decoded;
    uint32_t pitch = 0;
    DecodeLevel(desc, 0, 0, guest, nullptr, decoded, pitch);
    if (decoded.empty() || pitch < desc.width * 4)
        return Fail("surface upload", "render surface decoded to an empty or short image");
    std::vector<uint8_t> scaled(size_t(target.width) * target.height * 4);
    for (uint32_t y = 0; y < target.height; ++y) {
        const uint32_t sy = std::min(desc.height - 1, uint32_t(uint64_t(y) * desc.height / target.height));
        for (uint32_t x = 0; x < target.width; ++x) {
            const uint32_t sx = std::min(desc.width - 1, uint32_t(uint64_t(x) * desc.width / target.width));
            std::memcpy(scaled.data() + (size_t(y) * target.width + x) * 4,
                        decoded.data() + size_t(sy) * pitch + sx * 4, 4);
        }
    }
    std::string error;
    auto buffer = std::make_unique<vulkan::Buffer>();
    if (!buffer->Init(context_, scaled.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, error) ||
        !buffer->Write(0, scaled.data(), scaled.size(), error))
        return Fail("surface staging buffer", error);
    EndRendering();
    Transition(*target.image, target.layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, ColorAspect());
    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = {target.width, target.height, 1};
    auto copy_image = DeviceProc<PFN_vkCmdCopyBufferToImage>(context_, "vkCmdCopyBufferToImage");
    if (!copy_image) return Fail("surface upload command entry point");
    copy_image(ActiveFrame().command, buffer->Handle(), target.image->Handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    Transition(*target.image, target.layout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, ColorAspect());
    ActiveFrame().frame_buffers.push_back(std::move(buffer));
    target.content_generation = frame_generation_;
    target.content_epoch = DirtyEpoch(desc.data);
    target.content_signature = ContentSignature(desc);
    target.readback.clear();
    return true;
}

bool VulkanBackend::LatchCpuWrite(Target& target) {
    if (target.depth || !target.surface.data) return false;
    TextureDesc desc;
    if (!DescribeSurface(target.surface, desc)) return false;
    const uint64_t epoch = DirtyEpoch(desc.data);
    const uint64_t signature = ContentSignature(desc);
    {
        std::lock_guard lock(late_lock_);
        auto late_it = late_.find(desc.data);
        if (late_it != late_.end()) {
            Late& late = late_it->second;
            if (late.acknowledgement_pending && late.acknowledged_epoch == epoch &&
                late.acknowledged_signature == signature && late.acknowledged_identity == late.identity_generation &&
                late.image_identity == target.image->Handle() && SameLateDesc(late.desc, desc)) {
                late.acknowledgement_pending = false;
                target.content_generation = frame_generation_;
                target.content_epoch = epoch;
                target.content_signature = signature;
                target.readback.clear();
                return false;
            }
        }
    }
    if (target.content_generation == frame_generation_ && epoch <= target.content_epoch) return false;
    bool changed = epoch > target.content_epoch || signature != target.content_signature;
    // Sampling is sufficient for large textures in the ordinary path, but a previous synchronous readback gives
    // us an exact packed comparison for the small CPU edits that often accompany movie/font updates.
    if (!changed && !target.readback.empty() && Format(desc.format).host == Host::BGRA8) {
        std::vector<uint8_t> decoded;
        uint32_t pitch = 0;
        DecodeLevel(desc, 0, 0, static_cast<const uint8_t*>(kernel::HostPtr(desc.data)), nullptr, decoded, pitch);
        if (pitch >= desc.width * 4 && decoded.size() >= size_t(desc.height) * pitch &&
            target.readback.size() == size_t(desc.width) * desc.height * 4) {
            for (uint32_t y = 0; y < desc.height && !changed; ++y)
                changed = std::memcmp(decoded.data() + size_t(y) * pitch,
                                      target.readback.data() + size_t(y) * desc.width * 4, desc.width * 4) != 0;
        }
    }
    target.content_generation = frame_generation_;
    target.content_epoch = epoch;
    target.content_signature = signature;
    if (!changed) return false;
    target.drawn = false;
    target.readback.clear();
    return true;
}

bool VulkanBackend::RecordTargetCopy(Target& source, Target& destination, uint32_t count, uint32_t src_rects,
                                     uint32_t dst_points) {
    if (!source.image || !destination.image || source.depth || destination.depth) return false;
    const bool listed = count && src_rects;
    std::vector<vulkan::CopyRect> source_rects;
    if (listed) {
        source_rects.resize(count);
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t at = src_rects + i * 16;
            source_rects[i] = {int32_t(kernel::R32(at)), int32_t(kernel::R32(at + 4)), int32_t(kernel::R32(at + 8)),
                               int32_t(kernel::R32(at + 12))};
        }
    }
    std::vector<vulkan::CopyPoint> destination_points;
    if (dst_points) {
        destination_points.resize(listed ? count : 1);
        for (size_t i = 0; i < destination_points.size(); ++i) {
            const uint32_t at = dst_points + uint32_t(i) * 8;
            destination_points[i] = {int32_t(kernel::R32(at)), int32_t(kernel::R32(at + 4))};
        }
    }
    std::vector<vulkan::ImageCopyRegion> regions;
    if (!vulkan::BuildImageCopyRegions(source.guest_width, source.guest_height, source.width, source.height,
                                       destination.guest_width, destination.guest_height, destination.width,
                                       destination.height, count,
                                       source_rects.empty() ? nullptr : source_rects.data(),
                                       destination_points.empty() ? nullptr : destination_points.data(), regions))
        return false;
    if (regions.empty()) return true;
    Transition(source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    Transition(destination, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    std::string error;
    if (!vulkan::RecordImageTransfer(context_, ActiveFrame().command, source.image->Handle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                     destination.image->Handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, kColorFormat,
                                     regions, error))
        return Fail("copy command", error);
    Transition(source, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    Transition(destination, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    destination.drawn = true;
    destination.drawn_generation = frame_generation_;
    destination.readback.clear();
    return true;
}

bool VulkanBackend::EnsureTarget(const Surface& source, bool depth, Target*& out) {
    out = nullptr;
    if (!source.data) return true;
    // Surface headers carry physical addresses; locks and texture descriptions use guest virtual addresses.
    Surface surface = source;
    surface.data = mem::VirtualFromPhysical(source.data);
    uint32_t guest_width = 0, guest_height = 0;
    SurfaceSize(surface, guest_width, guest_height);
    if (!guest_width || !guest_height) return Fail("target size", "surface has zero extent");
    uint32_t width = std::max(1u, uint32_t(std::lround(float(guest_width) * render_scale_)));
    uint32_t height = std::max(1u, uint32_t(std::lround(float(guest_height) * render_scale_)));
    uint64_t key = SurfaceKey(surface, depth);
    auto publish_late_identity = [&](Target& target) {
        if (depth || !target.image) return;
        TextureDesc late_desc;
        if (!DescribeSurface(surface, late_desc)) return;
        std::lock_guard lock(late_lock_);
        Late& late = late_[surface.data];
        if (!late.configured || !SameLateDesc(late.desc, late_desc) || late.width != target.width ||
            late.height != target.height || late.image_identity != target.image->Handle()) {
            late.desc = late_desc;
            late.width = target.width;
            late.height = target.height;
            late.image_identity = target.image->Handle();
            late.configured = true;
            ++late.identity_generation;
            late.retire_pending = true;
            late.acknowledgement_pending = false;
        }
    };
    auto it = targets_.find(key);
    if (it != targets_.end()) {
        Target& target = it->second;
        const bool changed = target.width != width || target.height != height || target.guest_width != guest_width ||
                             target.guest_height != guest_height || target.surface.format != surface.format ||
                             target.surface.size != surface.size;
        if (changed) {
            auto image = std::make_unique<vulkan::Image>();
            std::string recreate_error;
            VkFormat recreate_format = depth ? kDepthFormat : kColorFormat;
            VkImageUsageFlags recreate_usage = depth ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                                           VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
                                                     : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                                           VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            if (!image->Init(context_, width, height, recreate_format, recreate_usage,
                             depth ? DepthAspect() : ColorAspect(), VK_SAMPLE_COUNT_1_BIT, recreate_error))
                return Fail("target recreation", recreate_error);
            if (target.image) ActiveFrame().frame_images.push_back(std::move(target.image));
            target.image = std::move(image);
            target.surface = surface;
            target.guest_width = guest_width;
            target.guest_height = guest_height;
            target.width = width;
            target.height = height;
            target.layout = VK_IMAGE_LAYOUT_UNDEFINED;
            target.drawn = false;
            if (!depth && !UploadSurface(target)) return false;
        }
        // Surface headers rotate between stable guest allocations at SwapNow. Keep the image for the same data/layout
        // identity, but retain the current descriptor for pitch/format-sensitive readback and future checks.
        target.surface = surface;
        if (!depth && LatchCpuWrite(target) && !UploadSurface(target)) return false;
        publish_late_identity(target);
        out = &target;
        return true;
    }
    auto image = std::make_unique<vulkan::Image>();
    std::string error;
    VkFormat format = depth ? kDepthFormat : kColorFormat;
    VkImageUsageFlags usage = depth ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                           VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
                                    : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                          VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    VkImageAspectFlags aspect = depth ? DepthAspect() : ColorAspect();
    if (!image->Init(context_, width, height, format, usage, aspect, VK_SAMPLE_COUNT_1_BIT, error))
        return Fail("target image", error);
    Target target;
    target.image = std::move(image);
    target.surface = surface;
    target.guest_width = guest_width;
    target.guest_height = guest_height;
    target.width = width;
    target.height = height;
    target.depth = depth;
    auto inserted = targets_.emplace(key, std::move(target));
    out = &inserted.first->second;
    publish_late_identity(*out);
    if (!depth && !UploadSurface(*out)) return false;
    return true;
}

bool VulkanBackend::CompatibleTarget(const Target& target, const TextureDesc& desc, bool depth) const {
    if (target.depth != depth || !target.image || desc.cube || desc.depth != 1 || desc.levels == 0)
        return false;
    const FormatInfo& format = Format(desc.format);
    if (depth ? !format.depth : (format.host != Host::BGRA8 || format.compressed)) return false;
    TextureDesc surface_desc;
    if (!DescribeSurface(target.surface, surface_desc)) return false;
    return surface_desc.data == desc.data && surface_desc.format == desc.format &&
           surface_desc.width == desc.width && surface_desc.height == desc.height && surface_desc.depth == desc.depth &&
           surface_desc.cube == desc.cube && surface_desc.pitch == desc.pitch && surface_desc.linear == desc.linear &&
           target.guest_width == desc.width && target.guest_height == desc.height;
}

bool VulkanBackend::PrepareTargetForSampling(Target& target) {
    if (!LatchCpuWrite(target)) return true;
    // A CPU write can arrive after the previous draw left this target as the active dynamic-rendering attachment.
    // Close that scope before UploadSurface records its transfer transition and copy.
    EndRendering();
    return UploadSurface(target);
}

bool VulkanBackend::EnsureTexture(const State& state, int stage, Texture*& out, TextureDesc& desc) {
    out = nullptr;
    if (stage < 0 || stage >= 4 || !state.textures[size_t(stage)])
        return true;
    if (!Describe(state.textures[size_t(stage)], desc))
        return Fail("texture bind", "bound texture description is invalid");
    const FormatInfo& format = Format(desc.format);
    if (format.host == Host::None || !desc.width || !desc.height || !desc.depth || !desc.levels)
        return Fail("texture bind", "unsupported texture");
    PaletteInfo palette;
    ReadPalette(state, stage, desc, palette);
    // Texture headers are short-lived guest objects. The underlying data and palette identify the resource, matching
    // the D3D12 path and preventing a new device image for every header alias created by the title.
    uint64_t key = (uint64_t(palette.va) << 32) | desc.data;
    auto it = textures_.find(key);
    const uint64_t epoch = DirtyEpoch(desc.data);
    const bool depth = format.depth;
    auto target_it = targets_.find(SurfaceKey(Surface{.data = desc.data}, depth));
    Target* target = target_it != targets_.end() && CompatibleTarget(target_it->second, desc, depth)
                         ? &target_it->second
                         : nullptr;
    const bool target_cpu_changed = target && LatchCpuWrite(*target);
    if (target_cpu_changed) {
        EndRendering();
        if (!UploadSurface(*target)) return false;
    }
    const auto identify = [&] {
        return remaster::Identify(desc, static_cast<const uint8_t*>(kernel::HostPtr(desc.data)), palette.colours);
    };
    if (target) {
        // The target image is authoritative for this compatible address. Keep a lightweight texture record for
        // content identity/overrides, but retire any old guest-upload image instead of uploading stale guest bytes.
        const bool new_identity = it == textures_.end();
        if (new_identity) {
            Texture texture;
            texture.desc = desc;
            texture.id = identify();
            texture.epoch = epoch;
            texture.content_generation = frame_generation_;
            texture.content_signature = ContentSignature(desc) ^ epoch ^ palette.hash;
            texture.palette_va = palette.va;
            texture.palette_bytes = palette.bytes;
            texture.palette_hash = palette.hash;
            it = textures_.emplace(key, std::move(texture)).first;
        } else {
            Texture& texture = it->second;
            const bool descriptor_changed = texture.desc.data != desc.data || texture.desc.width != desc.width ||
                                            texture.desc.height != desc.height || texture.desc.depth != desc.depth ||
                                            texture.desc.format != desc.format || texture.desc.cube != desc.cube ||
                                            texture.desc.levels != desc.levels || texture.desc.pitch != desc.pitch ||
                                            texture.desc.linear != desc.linear;
            const bool palette_changed = texture.palette_va != palette.va || texture.palette_bytes != palette.bytes ||
                                         texture.palette_hash != palette.hash;
            if (texture.image) ActiveFrame().frame_images.push_back(std::move(texture.image));
            texture.desc = desc;
            texture.layout = VK_IMAGE_LAYOUT_UNDEFINED;
            if (descriptor_changed || palette_changed || target_cpu_changed || !texture.id) texture.id = identify();
            texture.epoch = epoch;
            texture.content_generation = frame_generation_;
            texture.content_signature = ContentSignature(desc) ^ epoch ^ palette.hash;
            texture.palette_va = palette.va;
            texture.palette_bytes = palette.bytes;
            texture.palette_hash = palette.hash;
        }
        out = &it->second;
        return true;
    }

    auto make_image = [&](std::unique_ptr<vulkan::Image>& image, std::string& error) {
        VkFormat vk_format = format.depth ? VK_FORMAT_D32_SFLOAT
                            : format.host == Host::BC1 ? VK_FORMAT_BC1_RGBA_UNORM_BLOCK
                            : format.host == Host::BC2 ? VK_FORMAT_BC2_UNORM_BLOCK
                            : format.host == Host::BC3 ? VK_FORMAT_BC3_UNORM_BLOCK
                            : format.host == Host::R32F ? VK_FORMAT_R32_SFLOAT
                                                         : kColorFormat;
        image = std::make_unique<vulkan::Image>();
        const VkImageViewType view_type = desc.cube ? VK_IMAGE_VIEW_TYPE_CUBE
                                         : desc.depth > 1 ? VK_IMAGE_VIEW_TYPE_3D : VK_IMAGE_VIEW_TYPE_2D;
        const uint32_t layers = desc.cube ? 6 : 1;
        return image->Init(context_, desc.width, desc.height, vk_format,
                           VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                           format.depth ? VK_IMAGE_ASPECT_DEPTH_BIT : ColorAspect(), VK_SAMPLE_COUNT_1_BIT, error,
                           desc.depth, layers, desc.levels, view_type);
    };
    if (it == textures_.end()) {
        Texture texture;
        texture.desc = desc;
        std::string error;
        if (!make_image(texture.image, error)) return Fail("texture image", error);
        it = textures_.emplace(key, std::move(texture)).first;
    }
    Texture& texture = it->second;
    const bool descriptor_changed = texture.desc.data != desc.data || texture.desc.width != desc.width ||
                                    texture.desc.height != desc.height || texture.desc.depth != desc.depth ||
                                    texture.desc.format != desc.format || texture.desc.cube != desc.cube ||
                                    texture.desc.levels != desc.levels || texture.desc.pitch != desc.pitch ||
                                    texture.desc.linear != desc.linear;
    if (!texture.image || descriptor_changed) {
        std::unique_ptr<vulkan::Image> image;
        std::string recreate_error;
        if (!make_image(image, recreate_error)) return Fail("texture recreation", recreate_error);
        if (texture.image) ActiveFrame().frame_images.push_back(std::move(texture.image));
        texture.image = std::move(image);
        texture.desc = desc;
        texture.id = 0;
        texture.layout = VK_IMAGE_LAYOUT_UNDEFINED;
        texture.epoch = 0;
        texture.content_generation = 0;
        texture.content_signature = 0;
        texture.palette_va = 0;
        texture.palette_bytes = 0;
        texture.palette_hash = 0;
    }
    // A static resource is valid with no lock epoch. The image still needs its first upload, and a recreated image
    // similarly starts in UNDEFINED even when the guest address has never been marked dirty.
    const bool palette_changed = texture.palette_va != palette.va || texture.palette_bytes != palette.bytes ||
                                 texture.palette_hash != palette.hash;
    bool content_changed = false;
    if (texture.content_generation != frame_generation_ || palette_changed || texture.layout == VK_IMAGE_LAYOUT_UNDEFINED) {
        const uint64_t signature = ContentSignature(desc) ^ epoch ^ palette.hash;
        content_changed = texture.content_generation != 0 && texture.content_signature != signature;
        texture.content_generation = frame_generation_;
        texture.content_signature = signature;
    }
    if (texture.layout == VK_IMAGE_LAYOUT_UNDEFINED || content_changed || texture.epoch != epoch || palette_changed) {
        if (!UploadGuestImage(*texture.image, texture.layout, desc, palette.colours)) return false;
        // Content identity is an upload-time property; avoid hashing the same guest bytes again for every draw.
        texture.id = identify();
        texture.epoch = epoch;
        texture.palette_va = palette.va;
        texture.palette_bytes = palette.bytes;
        texture.palette_hash = palette.hash;
    }
    out = &texture;
    return true;
}

bool VulkanBackend::SnapshotTargetForSampling(Target& target, VkImageView& view) {
    view = VK_NULL_HANDLE;
    if (!target.image || !target.width || !target.height) return Fail("target feedback", "invalid target snapshot request");
    // The source may be the attachment of the preceding draw in this command buffer. End that rendering scope before
    // making it a transfer source, then leave the original target in transfer-source state for BeginRendering's
    // attachment transition on the draw that follows.
    EndRendering();
    auto snapshot = std::make_unique<vulkan::Image>();
    std::string error;
    const VkFormat format = target.depth ? kDepthFormat : kColorFormat;
    const VkImageAspectFlags copy_aspect = target.depth ? VK_IMAGE_ASPECT_DEPTH_BIT : ColorAspect();
    const VkImageAspectFlags layout_aspect = target.depth ? DepthAspect() : ColorAspect();
    if (!snapshot->Init(context_, target.width, target.height, format,
                        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                        layout_aspect, VK_SAMPLE_COUNT_1_BIT, error))
        return Fail("target feedback image", error);
    VkImageLayout snapshot_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    Transition(target, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    Transition(*snapshot, snapshot_layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, layout_aspect);
    std::vector<vulkan::ImageCopyRegion> regions(1);
    regions[0].src_extent = {target.width, target.height, 1};
    regions[0].dst_extent = {target.width, target.height, 1};
    if (!vulkan::RecordImageTransfer(context_, ActiveFrame().command, target.image->Handle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                     snapshot->Handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, format, regions, error,
                                     copy_aspect))
        return Fail("target feedback copy", error);
    Transition(*snapshot, snapshot_layout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, layout_aspect);
    view = snapshot->SampledView();
    ActiveFrame().frame_images.push_back(std::move(snapshot));
    return true;
}

void VulkanBackend::Transition(vulkan::Image& image, VkImageLayout& current, VkImageLayout layout,
                               VkImageAspectFlags aspect) {
    if (current == layout) return;
    // Image layout barriers are not legal inside an active dynamic-rendering instance. Uploads and other
    // asynchronous image users can arrive after a draw, so the transition owns closing that rendering scope.
    EndRendering();
    auto barrier = draw_dispatch_.pipeline_barrier;
    if (!barrier) {
        Fail("image transition", "vkCmdPipelineBarrier unavailable");
        return;
    }
    VkPipelineStageFlags src, dst;
    VkAccessFlags src_access, dst_access;
    LayoutMasks(current, src, src_access);
    LayoutMasks(layout, dst, dst_access);
    VkImageMemoryBarrier info{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    info.srcAccessMask = src_access;
    info.dstAccessMask = dst_access;
    info.oldLayout = current;
    info.newLayout = layout;
    info.srcQueueFamilyIndex = info.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    info.image = image.Handle();
    info.subresourceRange = {aspect, 0, image.MipLevels(), 0, image.ArrayLayers()};
    barrier(ActiveFrame().command, src, dst, 0, 0, nullptr, 0, nullptr, 1, &info);
    current = layout;
}

void VulkanBackend::Transition(VkImage image, VkImageAspectFlags aspect, VkImageLayout& current,
                               VkImageLayout layout) {
    if (current == layout) return;
    EndRendering();
    auto barrier = draw_dispatch_.pipeline_barrier;
    if (!barrier) {
        Fail("external image transition", "vkCmdPipelineBarrier unavailable");
        return;
    }
    VkPipelineStageFlags src, dst;
    VkAccessFlags src_access, dst_access;
    LayoutMasks(current, src, src_access);
    LayoutMasks(layout, dst, dst_access);
    VkImageMemoryBarrier info{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    info.srcAccessMask = src_access;
    info.dstAccessMask = dst_access;
    info.oldLayout = current;
    info.newLayout = layout;
    info.srcQueueFamilyIndex = info.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    info.image = image;
    info.subresourceRange = {aspect, 0, 1, 0, 1};
    barrier(ActiveFrame().command, src, dst, 0, 0, nullptr, 0, nullptr, 1, &info);
    current = layout;
}

void VulkanBackend::TransitionSwapchainColorAttachment(VkImage image, VkImageLayout& current) {
    if (current == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) return;
    if (current != VK_IMAGE_LAYOUT_UNDEFINED && current != VK_IMAGE_LAYOUT_PRESENT_SRC_KHR) {
        Transition(image, ColorAspect(), current, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        return;
    }
    EndRendering();
    auto barrier = draw_dispatch_.pipeline_barrier;
    if (!barrier) {
        Fail("swapchain color transition", "vkCmdPipelineBarrier unavailable");
        return;
    }
    VkImageMemoryBarrier info{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    info.srcAccessMask = 0;
    info.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    info.oldLayout = current;
    info.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    info.srcQueueFamilyIndex = info.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    info.image = image;
    info.subresourceRange = {ColorAspect(), 0, 1, 0, 1};
    barrier(ActiveFrame().command, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
            0, nullptr, 0, nullptr, 1, &info);
    current = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
}

void VulkanBackend::Transition(Target& target, VkImageLayout layout) {
    Transition(*target.image, target.layout, layout, target.depth ? DepthAspect() : ColorAspect());
}

VkSampler VulkanBackend::SamplerFor(const State& state, int stage, bool shadow, VkFormat sampled_format) {
    const auto& tss = state.tss[size_t(stage)];
    SamplerKey key;
    key.mag = tss[kTssMagFilter];
    key.min = tss[kTssMinFilter];
    key.mip = tss[kTssMipFilter];
    key.address_u = tss[kTssAddressU];
    key.address_v = tss[kTssAddressV];
    key.address_w = tss[kTssAddressW];
    key.mip_lod_bias = tss[kTssMipLodBias];
    key.max_mip_level = tss[kTssMaxMipLevel];
    key.max_anisotropy = tss[kTssMaxAnisotropy];
    key.border_color = tss[kTssBorderColor];
    key.compare_enable = shadow;
    key.compare_op = shadow ? vulkan::ShadowCompareOp(state.rs[kRsShadowFunc]) : VK_COMPARE_OP_ALWAYS;
    const bool border = key.address_u == 4 || key.address_v == 4 || key.address_w == 4;
    const bool standard_border = key.border_color == 0 || key.border_color == 0xFF000000 || key.border_color == 0xFFFFFFFF;
    const bool typed_custom_border = border && !standard_border &&
                                     (!context_.CustomBorderColorWithoutFormatEnabled() ||
                                      context_.CustomBorderColorFormatRequiresTyped(sampled_format));
    if (border && !standard_border)
        key.sampled_format = typed_custom_border ? sampled_format : VK_FORMAT_UNDEFINED;
    if (auto found = stage_samplers_.find(key); found != stage_samplers_.end()) return found->second;

    const bool mirror_clamp = key.address_u == 5 || key.address_v == 5 || key.address_w == 5;
    if (mirror_clamp && !context_.SamplerMirrorClampToEdgeEnabled()) {
        Fail("stage sampler", "D3D mirror-once address mode requires VK_KHR_sampler_mirror_clamp_to_edge");
        return VK_NULL_HANDLE;
    }
    if (border && !standard_border) {
        if (!context_.CustomBorderColorEnabled()) {
            Fail("stage sampler", "custom D3D border color requires VK_EXT_custom_border_color");
            return VK_NULL_HANDLE;
        }
        if (!context_.MaxCustomBorderColorSamplers() ||
            custom_border_sampler_count_ >= context_.MaxCustomBorderColorSamplers()) {
            Fail("stage sampler", "custom D3D border color sampler limit exhausted");
            return VK_NULL_HANDLE;
        }
        if (typed_custom_border &&
            !context_.CustomBorderColorFormatSupported(sampled_format)) {
            Fail("stage sampler", "custom D3D border color format is not supported for sampled images");
            return VK_NULL_HANDLE;
        }
    }
    auto create = DeviceProc<PFN_vkCreateSampler>(context_, "vkCreateSampler");
    if (!create) {
        Fail("stage sampler", "vkCreateSampler unavailable");
        return VK_NULL_HANDLE;
    }
    VkSamplerCreateInfo info = vulkan::MakeSamplerCreateInfo(
        tss, context_.Properties().limits.maxSamplerAnisotropy, context_.Properties().limits.maxSamplerLodBias);
    vulkan::SamplerCustomBorderColorInfo custom_border_info;
    if (border && !standard_border) {
        custom_border_info = vulkan::MakeSamplerCustomBorderColorInfo(
            key.border_color, typed_custom_border ? sampled_format : VK_FORMAT_UNDEFINED);
        info.borderColor = VK_BORDER_COLOR_FLOAT_CUSTOM_EXT;
        info.pNext = &custom_border_info.extension;
    }
    info.compareEnable = key.compare_enable ? VK_TRUE : VK_FALSE;
    info.compareOp = static_cast<VkCompareOp>(key.compare_op);
    VkSampler sampler = VK_NULL_HANDLE;
    std::string error;
    if (!Result(create(context_.Device(), &info, nullptr, &sampler), "vkCreateSampler", error)) {
        Fail("stage sampler", error);
        return VK_NULL_HANDLE;
    }
    stage_samplers_.emplace(key, sampler);
    samplers_.push_back(sampler);
    if (border && !standard_border) ++custom_border_sampler_count_;
    return sampler;
}

VkPrimitiveTopology VulkanBackend::Topology(const Assembled& assembled) const {
    if (assembled.topology == Topology::Points) return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    if (assembled.topology == Topology::Lines)
        return assembled.strip ? VK_PRIMITIVE_TOPOLOGY_LINE_STRIP : VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    return assembled.strip ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
}

bool VulkanBackend::BeginRendering(Target* color, Target* depth, VkImageView color_view, VkFormat color_format,
                                   VkExtent2D extent) {
    EndRendering();
    if (color) {
        Transition(*color, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        color_view = color->image->View();
        color_format = kColorFormat;
        extent = {color->width, color->height};
    }
    if (depth) {
        Transition(*depth, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        if (!extent.width) extent = {depth->width, depth->height};
    }
    auto begin = draw_dispatch_.begin_rendering;
    if (!begin || !extent.width || !extent.height || (!color_view && !depth))
        return Fail("rendering begin", "invalid dynamic-rendering attachment");
    VkRenderingAttachmentInfo color_attachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    VkRenderingAttachmentInfo depth_attachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    VkRenderingInfo info{VK_STRUCTURE_TYPE_RENDERING_INFO};
    info.renderArea = {{0, 0}, extent};
    info.layerCount = 1;
    if (color_view) {
        color_attachment.imageView = color_view;
        color_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        info.colorAttachmentCount = 1;
        info.pColorAttachments = &color_attachment;
    }
    if (depth) {
        depth_attachment.imageView = depth->image->View();
        depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        info.pDepthAttachment = &depth_attachment;
        info.pStencilAttachment = &depth_attachment;
    }
    begin(ActiveFrame().command, &info);
    rendering_ = true;
    rendering_color_ = color;
    rendering_depth_ = depth;
    (void)color_format;
    return true;
}

void VulkanBackend::EndRendering() {
    if (!rendering_) return;
    if (draw_dispatch_.end_rendering) draw_dispatch_.end_rendering(ActiveFrame().command);
    else Fail("rendering end", "vkCmdEndRendering unavailable");
    rendering_ = false;
    rendering_color_ = rendering_depth_ = nullptr;
}

bool VulkanBackend::Rects(const State& state, Target& target, uint32_t count, const uint8_t* rects,
                          std::vector<VkClearRect>& out) {
    out.clear();
    auto add = [&](int32_t x0, int32_t y0, int32_t x1, int32_t y1) {
        float sx = float(target.width) / float(target.guest_width);
        float sy = float(target.height) / float(target.guest_height);
        int32_t left = std::clamp(int32_t(std::lround(x0 * sx)), 0, int32_t(target.width));
        int32_t top = std::clamp(int32_t(std::lround(y0 * sy)), 0, int32_t(target.height));
        int32_t right = std::clamp(int32_t(std::lround(x1 * sx)), 0, int32_t(target.width));
        int32_t bottom = std::clamp(int32_t(std::lround(y1 * sy)), 0, int32_t(target.height));
        if (right > left && bottom > top) out.push_back({{{left, top}, {uint32_t(right - left), uint32_t(bottom - top)}}, 0, 1});
    };
    if (count && rects) {
        for (uint32_t i = 0; i < count; ++i) {
            int32_t values[4];
            std::memcpy(values, rects + size_t(i) * sizeof(values), sizeof(values));
            add(values[0], values[1], values[2], values[3]);
        }
    } else {
        add(int32_t(state.viewport.x), int32_t(state.viewport.y), int32_t(state.viewport.x + state.viewport.w),
            int32_t(state.viewport.y + state.viewport.h));
    }
    return true;
}

bool VulkanBackend::CompileModule(const std::string& source, const char* entry, vulkan::ShaderStage stage,
                                  vulkan::ShaderModule& module, VkShaderModule& shader) {
    shader = VK_NULL_HANDLE;
    std::string error;
    if (!vulkan::CompileShader(source, entry, stage, module, &shader_cache_, error)) return Fail("shader compilation", error);
    auto create = DeviceProc<PFN_vkCreateShaderModule>(context_, "vkCreateShaderModule");
    if (!create) return Fail("shader module", "vkCreateShaderModule unavailable");
    VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    info.codeSize = module.spirv.size() * sizeof(uint32_t);
    info.pCode = module.spirv.data();
    if (!Result(create(context_.Device(), &info, nullptr, &shader), "vkCreateShaderModule", error))
        return Fail("shader module", error);
    if (perf_enabled_) ++ActiveFrame().perf_current_shader_modules;
    return true;
}

bool VulkanBackend::CreateBlitPipeline() {
    static const std::string source_vs = vulkan::kBlitVertexShader;
    static const std::string source_ps = vulkan::kBlitFragmentShader;
    auto destroy_shader = DeviceProc<PFN_vkDestroyShaderModule>(context_, "vkDestroyShaderModule");
    auto create = DeviceProc<PFN_vkCreateGraphicsPipelines>(context_, "vkCreateGraphicsPipelines");
    if (!create || !destroy_shader) return Fail("blit pipeline", "pipeline entry points unavailable");
    vulkan::ShaderModule vs, ps;
    VkShaderModule vsm = VK_NULL_HANDLE, psm = VK_NULL_HANDLE;
    if (!CompileModule(source_vs, "main", vulkan::ShaderStage::Vertex, vs, vsm)) return false;
    if (!CompileModule(source_ps, "main", vulkan::ShaderStage::Pixel, ps, psm)) {
        destroy_shader(context_.Device(), vsm, nullptr);
        return false;
    }
    VkPipelineShaderStageCreateInfo stages[2] = {
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vsm, "main", nullptr},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, psm, "main", nullptr},
    };
    VkPipelineVertexInputStateCreateInfo vertex_input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = viewport.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState attachment{};
    attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                                 VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = 1;
    blend.pAttachments = &attachment;
    const VkDynamicState states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = std::size(states);
    dynamic.pDynamicStates = states;
    VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rendering.colorAttachmentCount = 1;
    const VkFormat format = swapchain_.Format();
    rendering.pColorAttachmentFormats = &format;
    VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    info.pNext = &rendering;
    info.stageCount = 2;
    info.pStages = stages;
    info.pVertexInputState = &vertex_input;
    info.pInputAssemblyState = &assembly;
    info.pViewportState = &viewport;
    info.pRasterizationState = &raster;
    info.pMultisampleState = &multisample;
    info.pColorBlendState = &blend;
    info.pDynamicState = &dynamic;
    info.layout = pipeline_layout_;
    std::string error;
    VkResult result = create(context_.Device(), context_.PipelineCacheHandle(), 1, &info, nullptr, &blit_pipeline_);
    destroy_shader(context_.Device(), vsm, nullptr);
    destroy_shader(context_.Device(), psm, nullptr);
    if (!Result(result, "vkCreateGraphicsPipelines blit", error)) return Fail("blit pipeline", error);
    return true;
}

VkPipeline MakePipelineState(const vulkan::Context& context, VkPipelineLayout layout, const State& state,
                             const Assembled& assembled, const vulkan::ShaderModule& vs, VkShaderModule vsm,
                             const vulkan::ShaderModule& ps, VkShaderModule psm, const vulkan::ShaderModule* gs,
                             VkShaderModule gsm, VkFormat color_format, VkFormat depth_format) {
    vulkan::PipelineStateDesc pipeline_state;
    pipeline_state.binding_count = assembled.buffer_count;
    for (uint32_t i = 0; i < assembled.buffer_count; ++i)
        pipeline_state.bindings[i] = {i, assembled.buffers[i].stride, VK_VERTEX_INPUT_RATE_VERTEX};
    for (uint32_t reg = 0; reg < 16; ++reg) {
        if (!(assembled.regs & (1u << reg))) continue;
        const VertexElement& element = assembled.elements[reg];
        const VkFormat format = vulkan::GuestVertexFormat(element.format);
        if (format == VK_FORMAT_UNDEFINED || pipeline_state.attribute_count == pipeline_state.attributes.size())
            return VK_NULL_HANDLE;
        pipeline_state.attributes[pipeline_state.attribute_count++] = {reg, element.buffer, format, element.offset};
    }
    pipeline_state.topology = assembled.topology == Topology::Points ? VK_PRIMITIVE_TOPOLOGY_POINT_LIST
                         : assembled.topology == Topology::Lines ? (assembled.strip ? VK_PRIMITIVE_TOPOLOGY_LINE_STRIP : VK_PRIMITIVE_TOPOLOGY_LINE_LIST)
                         : (assembled.strip ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    vulkan::GuestPipelineState guest_state;
    guest_state.cull_mode = state.rs[kRsCullMode];
    guest_state.triangles = assembled.topology == Topology::Triangles;
    guest_state.z_enable = state.rs[kRsZEnable] != 0;
    guest_state.z_write_enable = SimpleState(state, kMZWriteEnable) != 0;
    guest_state.z_func = SimpleState(state, kMZFunc);
    guest_state.stencil_enable = state.rs[kRsStencilEnable] != 0;
    guest_state.stencil_func = SimpleState(state, kMStencilFunc);
    guest_state.stencil_mask = SimpleState(state, kMStencilMask);
    guest_state.stencil_write_mask = SimpleState(state, kMStencilWriteMask);
    guest_state.stencil_fail = state.rs[kRsStencilFail];
    guest_state.stencil_zfail = SimpleState(state, kMStencilZFail);
    guest_state.stencil_pass = SimpleState(state, kMStencilPass);
    guest_state.color_write_enable = SimpleState(state, kMColorWriteEnable);
    guest_state.blend_enable = SimpleState(state, kMBlendEnable) != 0;
    guest_state.src_blend = SimpleState(state, kMSrcBlend);
    guest_state.dst_blend = SimpleState(state, kMDestBlend);
    guest_state.blend_op = SimpleState(state, kMBlendOp);
    guest_state.solid_offset_enable = SimpleState(state, kMSolidOffsetEnable) != 0;
    guest_state.depth_offset = Bits(SimpleState(state, kMDepthOffset));
    guest_state.slope_scale = Bits(SimpleState(state, kMSlopeScale));
    guest_state.zscale = state.zscale;
    guest_state.color_format = color_format;
    guest_state.depth_format = depth_format;
    guest_state.stencil_format = depth_format == kDepthFormat ? depth_format : VK_FORMAT_UNDEFINED;
    vulkan::TranslateGuestPipelineState(guest_state, pipeline_state);
    return vulkan::CreatePipelineState(context, layout, vs, vsm, ps, psm, gs, gsm, pipeline_state);
}

bool VulkanBackend::CreateUi() {
    std::lock_guard lock(app::UiLock());
    if (!ImGui_ImplVulkan_LoadFunctions(ImGuiVulkanFunction, &context_))
        return Fail("ImGui Vulkan function loading");

    auto create_pool = DeviceProc<PFN_vkCreateDescriptorPool>(context_, "vkCreateDescriptorPool");
    if (!create_pool) return Fail("ImGui Vulkan descriptor pool", "vkCreateDescriptorPool unavailable");
    VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 256};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool.maxSets = 256;
    pool.poolSizeCount = 1;
    pool.pPoolSizes = &size;
    std::string error;
    if (!Result(create_pool(context_.Device(), &pool, nullptr, &ui_pool_), "vkCreateDescriptorPool", error))
        return Fail("ImGui Vulkan descriptor pool", error);

    ui_color_format_ = swapchain_.Format();
    ImGui_ImplVulkan_InitInfo info{};
    info.Instance = context_.Instance();
    info.PhysicalDevice = context_.PhysicalDevice();
    info.Device = context_.Device();
    info.QueueFamily = context_.GraphicsQueueFamily();
    info.Queue = context_.GraphicsQueue();
    info.DescriptorPool = ui_pool_;
    info.MinImageCount = std::max(2u, swapchain_.ImageCount());
    info.ImageCount = info.MinImageCount;
    info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    info.UseDynamicRendering = true;
    info.PipelineRenderingCreateInfo = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR};
    info.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    info.PipelineRenderingCreateInfo.pColorAttachmentFormats = &ui_color_format_;
    if (!ImGui_ImplVulkan_Init(&info)) {
        auto destroy_pool = DeviceProc<PFN_vkDestroyDescriptorPool>(context_, "vkDestroyDescriptorPool");
        if (destroy_pool) destroy_pool(context_.Device(), ui_pool_, nullptr);
        ui_pool_ = VK_NULL_HANDLE;
        return Fail("ImGui Vulkan initialization");
    }
    ui_initialized_ = true;
    LOGI("vulkan: ImGui Vulkan UI initialized for %u swapchain images", swapchain_.ImageCount());
    return true;
}

bool VulkanBackend::RecreateUi() {
    std::lock_guard lock(app::UiLock());
    if (!ui_initialized_) return CreateUi();
    for (auto& [key, texture] : ui_textures_) {
        if (texture.descriptor) ImGui_ImplVulkan_RemoveTexture(texture.descriptor);
        texture.descriptor = VK_NULL_HANDLE;
    }
    ImGui_ImplVulkan_Shutdown();
    ui_initialized_ = false;
    ui_color_format_ = swapchain_.Format();
    ImGui_ImplVulkan_InitInfo info{};
    info.Instance = context_.Instance();
    info.PhysicalDevice = context_.PhysicalDevice();
    info.Device = context_.Device();
    info.QueueFamily = context_.GraphicsQueueFamily();
    info.Queue = context_.GraphicsQueue();
    info.DescriptorPool = ui_pool_;
    info.MinImageCount = std::max(2u, swapchain_.ImageCount());
    info.ImageCount = info.MinImageCount;
    info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    info.UseDynamicRendering = true;
    info.PipelineRenderingCreateInfo = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR};
    info.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    info.PipelineRenderingCreateInfo.pColorAttachmentFormats = &ui_color_format_;
    if (!ImGui_ImplVulkan_Init(&info)) return Fail("ImGui Vulkan recreation");
    ui_initialized_ = true;
    for (auto& [key, texture] : ui_textures_)
        if (texture.image) {
            texture.descriptor = ImGui_ImplVulkan_AddTexture(linear_sampler_, texture.image->View(),
                                                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            if (!texture.descriptor)
                LOGW("vulkan: ImGui could not recreate descriptor for UI image %s", key.c_str());
        }
    LOGI("vulkan: ImGui Vulkan UI recreated for format %d and %u swapchain images", int(ui_color_format_),
         swapchain_.ImageCount());
    return true;
}

bool VulkanBackend::PumpOverrides() {
    std::vector<TextureOverride> set;
    if (!TakeTextureOverrides(set)) return true;
    // The other frame slot may still sample the previous override image. Keep those images with the
    // slot being recorded; with two slots that lifetime covers the other in-flight submission as well.
    for (auto& [id, texture] : override_textures_)
        if (texture.image) ActiveFrame().frame_images.push_back(std::move(texture.image));
    override_textures_.clear();
    pending_override_uploads_.clear();
    for (const TextureOverride& override_texture : set) {
        const size_t bytes = size_t(override_texture.w) * override_texture.h * 4;
        if (!override_texture.id || !override_texture.w || !override_texture.h ||
            override_texture.rgba.size() < bytes)
            continue;
        OverrideTex texture;
        texture.image = std::make_unique<vulkan::Image>();
        std::string error;
        if (!texture.image->Init(context_, override_texture.w, override_texture.h, VK_FORMAT_R8G8B8A8_UNORM,
                                 VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, ColorAspect(),
                                 VK_SAMPLE_COUNT_1_BIT, error))
            return Fail("texture override image", error);
        auto staging = std::make_unique<vulkan::Buffer>();
        if (!staging->Init(context_, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, error) ||
            !staging->Write(0, override_texture.rgba.data(), bytes, error))
            return Fail("texture override staging", error);
        override_textures_[override_texture.id] = std::move(texture);
        pending_override_uploads_.push_back(
            {override_texture.id, std::move(staging), override_texture.w, override_texture.h});
    }
    if (!FlushOverrideUploads()) return false;
    LOGI("vulkan: %zu texture overrides active", override_textures_.size());
    return true;
}

bool VulkanBackend::FlushOverrideUploads() {
    if (pending_override_uploads_.empty()) return true;
    EndRendering();
    auto copy_image = DeviceProc<PFN_vkCmdCopyBufferToImage>(context_, "vkCmdCopyBufferToImage");
    if (!copy_image) return Fail("texture override upload", "vkCmdCopyBufferToImage unavailable");
    for (PendingOverrideUpload& pending : pending_override_uploads_) {
        auto it = override_textures_.find(pending.id);
        if (it == override_textures_.end() || !it->second.image || !pending.staging)
            return Fail("texture override upload", "override image disappeared");
        OverrideTex& texture = it->second;
        Transition(*texture.image, texture.layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, ColorAspect());
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {pending.width, pending.height, 1};
        copy_image(ActiveFrame().command, pending.staging->Handle(), texture.image->Handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                   &copy);
        Transition(*texture.image, texture.layout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, ColorAspect());
        ActiveFrame().frame_buffers.push_back(std::move(pending.staging));
    }
    pending_override_uploads_.clear();
    return true;
}

bool VulkanBackend::CreateDrawPipeline(const State& state, const Assembled& assembled, const std::string& vs_source,
                                       const std::string& ps_source, const std::string& gs_source, VkFormat color_format,
                                       VkFormat depth_format, VkPipeline& pipeline, PipelineKey& key) {
    key = {};
    key.vertex = Hash(vs_source.data(), vs_source.size());
    key.pixel = Hash(ps_source.data(), ps_source.size());
    key.geometry = Hash(gs_source.data(), gs_source.size());
    key.regs = assembled.regs;
    key.topology = uint8_t(assembled.topology);
    key.strip = assembled.strip;
    key.color = color_format;
    key.depth = depth_format;
    key.buffers = assembled.buffer_count;
    for (uint32_t r = 0; r < 16; ++r)
        if (assembled.regs & (1u << r)) key.elements[r] = assembled.elements[r].format | (uint32_t(assembled.elements[r].buffer) << 8) |
                                                                                     (uint32_t(assembled.elements[r].offset) << 16);
    for (uint32_t i = 0; i < assembled.buffer_count; ++i) key.strides[i] = assembled.buffers[i].stride;
    key.blend = SimpleState(state, kMBlendEnable);
    key.src = SimpleState(state, kMSrcBlend);
    key.dst = SimpleState(state, kMDestBlend);
    key.op = SimpleState(state, kMBlendOp);
    key.write_mask = SimpleState(state, kMColorWriteEnable);
    key.depth_enable = state.rs[kRsZEnable];
    key.depth_write = SimpleState(state, kMZWriteEnable);
    key.depth_func = SimpleState(state, kMZFunc);
    key.cull = state.rs[kRsCullMode];
    key.front_ccw = vulkan::RasterFrontFace(state.rs[kRsCullMode]) == VK_FRONT_FACE_COUNTER_CLOCKWISE;
    key.fill = state.rs[kRsFillMode];
    key.stencil_enable = state.rs[kRsStencilEnable];
    key.stencil_func = SimpleState(state, kMStencilFunc);
    key.stencil_mask = SimpleState(state, kMStencilMask) & 0xff;
    key.stencil_write_mask = SimpleState(state, kMStencilWriteMask) & 0xff;
    key.stencil_fail = state.rs[kRsStencilFail];
    key.stencil_zfail = SimpleState(state, kMStencilZFail);
    key.stencil_pass = SimpleState(state, kMStencilPass);
    if (assembled.topology == Topology::Triangles && SimpleState(state, kMSolidOffsetEnable)) {
        key.depth_bias = int32_t(std::lround(Bits(SimpleState(state, kMDepthOffset)) * kDepth24Max /
                                             std::max(state.zscale, 1.0f)));
        key.depth_slope_bits = SimpleState(state, kMSlopeScale);
    }
    if (key.fill == 0x1b01)
        return Fail("wireframe pipeline", "fillModeNonSolid is not enabled on this device");
    if (auto it = pipelines_.find(key); it != pipelines_.end()) {
        pipeline = it->second;
        return true;
    }
    if (perf_enabled_) ++ActiveFrame().perf_current_pipeline_misses;
    vulkan::ShaderModule vs, ps, gs;
    VkShaderModule vsm = VK_NULL_HANDLE, psm = VK_NULL_HANDLE, gsm = VK_NULL_HANDLE;
    auto destroy_shader = DeviceProc<PFN_vkDestroyShaderModule>(context_, "vkDestroyShaderModule");
    auto release_shaders = [&] {
        if (!destroy_shader) return;
        if (vsm) destroy_shader(context_.Device(), vsm, nullptr);
        if (psm) destroy_shader(context_.Device(), psm, nullptr);
        if (gsm) destroy_shader(context_.Device(), gsm, nullptr);
    };
    if (!CompileModule(vs_source, "main", vulkan::ShaderStage::Vertex, vs, vsm) ||
        !CompileModule(ps_source, "main", vulkan::ShaderStage::Pixel, ps, psm)) {
        release_shaders();
        return false;
    }
    const vulkan::ShaderModule* gs_ptr = nullptr;
    if (!gs_source.empty()) {
        if (!CompileModule(gs_source, "gs", vulkan::ShaderStage::Geometry, gs, gsm)) {
            release_shaders();
            return false;
        }
        gs_ptr = &gs;
    }
    pipeline = MakePipelineState(context_, pipeline_layout_, state, assembled, vs, vsm, ps, psm, gs_ptr, gsm,
                                 color_format, depth_format);
    release_shaders();
    if (!pipeline) return Fail("graphics pipeline", "vkCreateGraphicsPipelines failed for generated CoDF shaders");
    pipelines_.emplace(key, pipeline);
    return true;
}

void VulkanBackend::RetireUploads() {
    // Rewind only after the command batch fence has completed; earlier draws retain their bytes.
    ActiveFrame().uploads.Reset();
    if (perf_enabled_) ++perf_window_upload_rewinds_;
}

size_t VulkanBackend::UploadBlockIndex(const vulkan::Buffer* buffer) const {
    return ActiveFrame().uploads.BlockIndex(buffer);
}

bool VulkanBackend::AllocateUpload(VkDeviceSize size, VkDeviceSize alignment, vulkan::Buffer*& buffer,
                                  VkDeviceSize& offset) {
    std::string error;
    const size_t blocks_before = ActiveFrame().uploads.BlockCount();
    if (!ActiveFrame().uploads.Allocate(size, alignment, buffer, offset, error)) return Fail("draw upload", error);
    if (perf_enabled_) {
        ++ActiveFrame().perf_current_upload_allocations;
        ActiveFrame().perf_current_upload_bytes += size;
        if (ActiveFrame().uploads.BlockCount() > blocks_before) ++ActiveFrame().perf_current_upload_block_grows;
    }
    return true;
}

bool VulkanBackend::UploadDrawConstants(const State& state, const DrawConstants& constants, vulkan::Buffer*& buffer,
                                        std::array<uint32_t, vulkan::kDynamicConstantBindingCount>& offsets) {
    offsets.fill(0);
    const VkDeviceSize alignment = std::max<VkDeviceSize>(kConstantAlignment, context_.Properties().limits.minUniformBufferOffsetAlignment);
    const VkDeviceSize draw_relative = Align(kConstantRange, alignment);
    const VkDeviceSize size = draw_relative + kConstantRange;
    VkDeviceSize vertex_offset = 0;
    if (!AllocateUpload(size, alignment, buffer, vertex_offset)) return false;
    const VkDeviceSize draw_offset = vertex_offset + draw_relative;
    std::string error;
    if (!buffer->Write(vertex_offset, state.vsc.data(), sizeof(state.vsc), error) ||
        !buffer->Write(draw_offset, &constants, sizeof(constants), error))
        return Fail("constant upload", error);
    if (perf_enabled_ && perf_upload_trace_count_ < 12) {
        LOGR("vulkan: upload constants block=%zu vertex_offset=%llu draw_offset=%llu",
             UploadBlockIndex(buffer), (unsigned long long)vertex_offset, (unsigned long long)draw_offset);
        ++perf_upload_trace_count_;
    }
    offsets[0] = static_cast<uint32_t>(vertex_offset);
    offsets[1] = static_cast<uint32_t>(draw_offset);
    return true;
}

bool VulkanBackend::AllocateDescriptors(const std::array<VkImageView, vulkan::kTextureBindingCount>& textures,
                                        const std::array<VkSampler, vulkan::kSamplerBindingCount>& samplers,
                                        vulkan::Buffer& constants, VkDeviceSize range, vulkan::DescriptorSet& set) {
    vulkan::DescriptorResources resources;
    resources.constant_buffer = constants.Handle();
    resources.constant_buffer_size = constants.Size();
    resources.constant_range = range;
    resources.textures = textures;
    resources.samplers = samplers;
    std::string error;
    if (!ActiveFrame().descriptors.Allocate(resources, set, error)) return Fail("descriptor allocation", error);
    ActiveFrame().descriptors_used = true;
    if (perf_enabled_) ++ActiveFrame().perf_current_descriptor_requests;
    return true;
}

const vulkan::DrawSources& VulkanBackend::DrawSourcesFor(const State& state, const Assembled& assembled,
                                                         const PixelKey& pixel_key) {
    const bool programmable = (state.vs_handle & 1) && state.vs && !state.vs->program.empty();
    const FixedKey fixed = programmable ? FixedKey{} : MakeFixedKey(state);
    bool cache_hit = false;
    const vulkan::DrawSources& sources = shader_sources_.Get(
        programmable ? state.vs.get() : nullptr, state.vs_handle, assembled.regs, assembled.decode,
        assembled.pretransformed, fixed, pixel_key, assembled.topology == Topology::Points, &cache_hit);
    if (perf_enabled_) {
        if (cache_hit)
            ++ActiveFrame().perf_current_source_hits;
        else
            ++ActiveFrame().perf_current_source_misses;
    }
    return sources;
}

bool VulkanBackend::RecordDraw(const State& state, const Assembled& assembled, const PixelKey& pixel_key,
                               Target* color, Target* depth, const std::array<VkImageView, 4>& stage_views,
                               const std::array<VkSampler, 4>& stage_samplers, const DrawConstants& constants) {
    std::chrono::steady_clock::time_point detail_start;
    if (ActiveFrame().perf_detail_frame_active) detail_start = std::chrono::steady_clock::now();
    const vulkan::DrawSources& sources = DrawSourcesFor(state, assembled, pixel_key);
    VkFormat color_format = color ? kColorFormat : VK_FORMAT_UNDEFINED;
    VkFormat depth_format = depth ? kDepthFormat : VK_FORMAT_UNDEFINED;
    VkPipeline pipeline = VK_NULL_HANDLE;
    PipelineKey pipeline_key;
    if (!CreateDrawPipeline(state, assembled, sources.vertex, sources.pixel, sources.geometry, color_format, depth_format,
                            pipeline, pipeline_key)) {
        if (ActiveFrame().perf_detail_frame_active)
            ActiveFrame().perf_current_detail_pipeline_source_ns += ElapsedNanoseconds(detail_start);
        return false;
    }
    if (ActiveFrame().perf_detail_frame_active)
        ActiveFrame().perf_current_detail_pipeline_source_ns += ElapsedNanoseconds(detail_start);

    vulkan::Buffer* constants_buffer = nullptr;
    std::array<uint32_t, vulkan::kDynamicConstantBindingCount> dynamic_offsets{};
    if (ActiveFrame().perf_detail_frame_active) detail_start = std::chrono::steady_clock::now();
    if (!UploadDrawConstants(state, constants, constants_buffer, dynamic_offsets)) {
        if (ActiveFrame().perf_detail_frame_active)
            ActiveFrame().perf_current_detail_constant_vertex_index_upload_ns += ElapsedNanoseconds(detail_start);
        return false;
    }
    if (ActiveFrame().perf_detail_frame_active)
        ActiveFrame().perf_current_detail_constant_vertex_index_upload_ns += ElapsedNanoseconds(detail_start);
    std::array<VkImageView, vulkan::kTextureBindingCount> views{};
    views.fill(dummy_->View());
    std::array<VkSampler, vulkan::kSamplerBindingCount> samplers{};
    samplers.fill(linear_sampler_);
    for (int i = 0; i < 4; ++i) {
        if (stage_views[size_t(i)]) views[size_t(i)] = stage_views[size_t(i)];
        if (stage_samplers[size_t(i)]) samplers[size_t(i)] = stage_samplers[size_t(i)];
    }
    vulkan::DescriptorSet set;
    if (ActiveFrame().perf_detail_frame_active) detail_start = std::chrono::steady_clock::now();
    if (!AllocateDescriptors(views, samplers, *constants_buffer, kConstantRange, set)) {
        if (ActiveFrame().perf_detail_frame_active)
            ActiveFrame().perf_current_detail_descriptor_lookup_allocation_ns += ElapsedNanoseconds(detail_start);
        return false;
    }
    if (ActiveFrame().perf_detail_frame_active)
        ActiveFrame().perf_current_detail_descriptor_lookup_allocation_ns += ElapsedNanoseconds(detail_start);

    std::array<VkBuffer, 16> vertex_buffers{};
    std::array<VkDeviceSize, 16> vertex_offsets{};
    if (ActiveFrame().perf_detail_frame_active) detail_start = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < assembled.buffer_count; ++i) {
        const VertexBuffer& source = assembled.buffers[i];
        vulkan::Buffer* buffer = nullptr;
        VkDeviceSize offset = 0;
        std::vector<uint8_t> upload_data;
        const uint8_t* upload_source = source.src;
        if (recolor_ && recolor_element_.buffer == i && source.stride && source.src &&
            recolor_element_.format == DXGI_FORMAT_B8G8R8A8_UNORM) {
            upload_data.assign(source.src, source.src + source.bytes);
            for (uint32_t offset = recolor_element_.offset; offset + 4 <= source.bytes; offset += source.stride) {
                uint32_t value;
                std::memcpy(&value, upload_data.data() + offset, 4);
                value = (value & 0xFF000000u) | (recolor_argb_ & 0x00FFFFFFu);
                std::memcpy(upload_data.data() + offset, &value, 4);
            }
            upload_source = upload_data.data();
        }
        std::string error;
        if (!AllocateUpload(std::max<uint32_t>(source.bytes, 4u), 16, buffer, offset) ||
            !buffer->Write(offset, upload_source, source.bytes, error)) {
            if (ActiveFrame().perf_detail_frame_active)
                ActiveFrame().perf_current_detail_constant_vertex_index_upload_ns += ElapsedNanoseconds(detail_start);
            return Fail("vertex upload", error);
        }
        vertex_buffers[i] = buffer->Handle();
        vertex_offsets[i] = offset;
        if (perf_enabled_ && perf_upload_trace_count_ < 12) {
            LOGR("vulkan: upload vertex slot=%u block=%zu offset=%llu bytes=%u", i, UploadBlockIndex(buffer),
                 (unsigned long long)offset, source.bytes);
            ++perf_upload_trace_count_;
        }
    }
    vulkan::Buffer* index_buffer = nullptr;
    VkDeviceSize index_offset = 0;
    if (assembled.index_count) {
        const uint64_t bytes = uint64_t(assembled.index_count) * (assembled.index16 ? 2u : 4u);
        std::string error;
        if (!AllocateUpload(bytes, 4, index_buffer, index_offset) ||
            !index_buffer->Write(index_offset, assembled.index_src, size_t(bytes), error)) {
            if (ActiveFrame().perf_detail_frame_active)
                ActiveFrame().perf_current_detail_constant_vertex_index_upload_ns += ElapsedNanoseconds(detail_start);
            return Fail("index upload", error);
        }
        if (perf_enabled_ && perf_upload_trace_count_ < 12) {
            LOGR("vulkan: upload index block=%zu offset=%llu bytes=%llu", UploadBlockIndex(index_buffer),
                 (unsigned long long)index_offset, (unsigned long long)bytes);
            ++perf_upload_trace_count_;
        }
    }
    if (ActiveFrame().perf_detail_frame_active)
        ActiveFrame().perf_current_detail_constant_vertex_index_upload_ns += ElapsedNanoseconds(detail_start);

    if (ActiveFrame().perf_detail_frame_active) detail_start = std::chrono::steady_clock::now();
    if (!BeginRendering(color, depth)) {
        if (ActiveFrame().perf_detail_frame_active)
            ActiveFrame().perf_current_detail_command_recording_ns += ElapsedNanoseconds(detail_start);
        return false;
    }
    auto bind_pipeline = draw_dispatch_.bind_pipeline;
    auto bind_vertex = draw_dispatch_.bind_vertex_buffers;
    auto bind_index = draw_dispatch_.bind_index_buffer;
    auto draw = draw_dispatch_.draw;
    auto draw_indexed = draw_dispatch_.draw_indexed;
    auto set_viewport = draw_dispatch_.set_viewport;
    auto set_scissor = draw_dispatch_.set_scissor;
    auto set_blend = draw_dispatch_.set_blend_constants;
    auto set_stencil = draw_dispatch_.set_stencil_reference;
    if (!bind_pipeline || !bind_vertex || !bind_index || !draw || !draw_indexed || !set_viewport || !set_scissor ||
        !set_blend || !set_stencil) {
        if (ActiveFrame().perf_detail_frame_active)
            ActiveFrame().perf_current_detail_command_recording_ns += ElapsedNanoseconds(detail_start);
        return Fail("draw commands", "required Vulkan draw entry point unavailable");
    }
    bind_pipeline(ActiveFrame().command, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    if (!ActiveFrame().descriptors.Bind(ActiveFrame().command, pipeline_layout_, set, dynamic_offsets, last_error_))
    {
        if (ActiveFrame().perf_detail_frame_active)
            ActiveFrame().perf_current_detail_command_recording_ns += ElapsedNanoseconds(detail_start);
        return Fail("descriptor bind", last_error_);
    }
    Target* viewport_target = color ? color : depth;
    VkViewport viewport{0.0f, float(viewport_target->height), float(viewport_target->width), -float(viewport_target->height), 0.0f, 1.0f};
    set_viewport(ActiveFrame().command, 0, 1, &viewport);
    float scale_x = float(viewport_target->width) / float(viewport_target->guest_width);
    float scale_y = float(viewport_target->height) / float(viewport_target->guest_height);
    int32_t x0 = int32_t(state.viewport.x * scale_x), y0 = int32_t(state.viewport.y * scale_y);
    int32_t x1 = int32_t((state.viewport.x + state.viewport.w) * scale_x);
    int32_t y1 = int32_t((state.viewport.y + state.viewport.h) * scale_y);
    if (state.scissor_count) {
        x0 = std::max(x0, int32_t(state.scissors[0] * scale_x));
        y0 = std::max(y0, int32_t(state.scissors[1] * scale_y));
        x1 = std::min(x1, int32_t(state.scissors[2] * scale_x));
        y1 = std::min(y1, int32_t(state.scissors[3] * scale_y));
    }
    VkRect2D scissor{{std::clamp(x0, 0, int32_t(viewport_target->width)),
                      std::clamp(y0, 0, int32_t(viewport_target->height))},
                     {uint32_t(std::max(0, std::min(x1, int32_t(viewport_target->width)) - std::clamp(x0, 0, int32_t(viewport_target->width)))),
                      uint32_t(std::max(0, std::min(y1, int32_t(viewport_target->height)) - std::clamp(y0, 0, int32_t(viewport_target->height))))}};
    set_scissor(ActiveFrame().command, 0, 1, &scissor);
    float blend_color[4];
    Color(SimpleState(state, kMBlendColor), blend_color);
    set_blend(ActiveFrame().command, blend_color);
    set_stencil(ActiveFrame().command, VK_STENCIL_FACE_FRONT_AND_BACK, SimpleState(state, kMStencilRef) & 0xff);
    if (assembled.buffer_count) bind_vertex(ActiveFrame().command, 0, assembled.buffer_count, vertex_buffers.data(), vertex_offsets.data());
    if (assembled.index_count) {
        bind_index(ActiveFrame().command, index_buffer->Handle(), index_offset, assembled.index16 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);
        draw_indexed(ActiveFrame().command, assembled.index_count, 1, 0, assembled.base_vertex, 0);
    } else {
        draw(ActiveFrame().command, assembled.draw_count, 1, 0, 0);
    }
    if (color) {
        color->drawn = true;
        color->drawn_generation = frame_generation_;
    }
    if (depth) {
        depth->drawn = true;
        depth->drawn_generation = frame_generation_;
    }
    if (ActiveFrame().perf_detail_frame_active)
        ActiveFrame().perf_current_detail_command_recording_ns += ElapsedNanoseconds(detail_start);
    if (perf_enabled_) ++ActiveFrame().perf_current_draws;
    return true;
}

uint64_t VulkanBackend::UiImage(const std::string& key, const remaster::Image* source, bool source_missing) {
    auto existing = ui_textures_.find(key);
    if (existing != ui_textures_.end()) {
        return reinterpret_cast<uint64_t>(existing->second.descriptor);
    }
    UiTex texture;
    if (source_missing) {
        LOGW("vulkan: optional UI image missing %s", key.c_str());
        ui_textures_.emplace(key, std::move(texture));
        return 0;
    }
    if (!source || !source->w || !source->h || source->mips == 0 || source->offsets.empty() || source->data.empty()) {
        LOGW("vulkan: UI image data unavailable %s", key.c_str());
        ui_textures_.emplace(key, std::move(texture));
        return 0;
    }
    const VkFormat format = UiFormat(source->fmt);
    const uint32_t row_bytes = source->RowBytes(0);
    const uint32_t rows = source->Rows(0);
    const size_t offset = source->offsets[0];
    const size_t bytes = size_t(row_bytes) * rows;
    if (format == VK_FORMAT_UNDEFINED || !row_bytes || !rows || offset > source->data.size() ||
        bytes > source->data.size() - offset) {
        LOGW("vulkan: unsupported UI image format %d for %s", int(source->fmt), key.c_str());
        ui_textures_.emplace(key, std::move(texture));
        return 0;
    }
    texture.image = std::make_unique<vulkan::Image>();
    std::string error;
    if (!texture.image->Init(context_, source->w, source->h, format,
                             VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, ColorAspect(),
                             VK_SAMPLE_COUNT_1_BIT, error)) {
        LOGW("vulkan: UI image unavailable %s (format %d): %s", key.c_str(), int(source->fmt), error.c_str());
        ui_textures_.emplace(key, std::move(texture));
        return 0;
    }
    auto staging = std::make_unique<vulkan::Buffer>();
    if (!staging->Init(context_, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, error) ||
        !staging->Write(0, source->data.data() + offset, bytes, error)) {
        LOGW("vulkan: UI image staging unavailable %s: %s", key.c_str(), error.c_str());
        ui_textures_.emplace(key, std::move(texture));
        return 0;
    }
    const auto inserted = ui_textures_.emplace(key, std::move(texture));
    UiTex& stored = inserted.first->second;
    stored.descriptor = ImGui_ImplVulkan_AddTexture(linear_sampler_, stored.image->View(),
                                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    if (!stored.descriptor) {
        LOGW("vulkan: ImGui could not allocate a descriptor for UI image %s", key.c_str());
        return 0;
    }
    pending_ui_uploads_.push_back({key, std::move(staging), format, source->w, source->h});
    return reinterpret_cast<uint64_t>(stored.descriptor);
}

bool VulkanBackend::FlushUiUploads(const vulkan::AcquireResult& acquired) {
    if (pending_ui_uploads_.empty()) return true;
    EndRendering();
    auto copy_image = DeviceProc<PFN_vkCmdCopyBufferToImage>(context_, "vkCmdCopyBufferToImage");
    if (!copy_image) return Fail("UI image upload", "vkCmdCopyBufferToImage unavailable");
    for (PendingUiUpload& pending : pending_ui_uploads_) {
        auto it = ui_textures_.find(pending.key);
        if (it == ui_textures_.end() || !it->second.image || !pending.staging) return Fail("UI image upload", "UI image disappeared");
        UiTex& texture = it->second;
        Transition(*texture.image, texture.layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, ColorAspect());
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {pending.width, pending.height, 1};
        copy_image(ActiveFrame().command, pending.staging->Handle(), texture.image->Handle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                   &copy);
        Transition(*texture.image, texture.layout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, ColorAspect());
        ActiveFrame().frame_buffers.push_back(std::move(pending.staging));
    }
    pending_ui_uploads_.clear();
    return BeginRendering(nullptr, nullptr, swapchain_.View(acquired), swapchain_.Format(), swapchain_.Extent());
}

bool VulkanBackend::RenderUi(const vulkan::AcquireResult& acquired) {
    if (!ui_initialized_) return true;
    std::lock_guard lock(app::UiLock());
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();
    UiScope ui_scope(this);
    if (app::kGame.ui) app::kGame.ui(fps_);
    ImGui::Render();
    if (!FlushUiUploads(acquired)) return false;
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), ActiveFrame().command);
    return true;
}

bool VulkanBackend::RecordSwapchainScreenshot(VkImage image, VkImageLayout& layout) {
    if (!(swapchain_.ImageUsage() & VK_IMAGE_USAGE_TRANSFER_SRC_BIT))
        return Fail("screenshot", "swapchain does not support transfer-source readback");
    const VkFormat format = swapchain_.Format();
    const bool bgra = format == VK_FORMAT_B8G8R8A8_UNORM || format == VK_FORMAT_B8G8R8A8_SRGB;
    const bool rgba = format == VK_FORMAT_R8G8B8A8_UNORM || format == VK_FORMAT_R8G8B8A8_SRGB;
    if (!bgra && !rgba)
        return Fail("screenshot", "swapchain format is not a supported 32-bit BGRA/RGBA layout");
    const VkExtent2D extent = swapchain_.Extent();
    const VkDeviceSize bytes = VkDeviceSize(extent.width) * extent.height * 4;
    auto buffer = std::make_unique<vulkan::Buffer>();
    std::string error;
    if (!buffer->Init(context_, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, error))
        return Fail("screenshot buffer", error);
    auto copy_image = DeviceProc<PFN_vkCmdCopyImageToBuffer>(context_, "vkCmdCopyImageToBuffer");
    auto barrier = DeviceProc<PFN_vkCmdPipelineBarrier>(context_, "vkCmdPipelineBarrier");
    if (!copy_image || !barrier) return Fail("screenshot command", "required Vulkan readback entry point unavailable");
    Transition(image, ColorAspect(), layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = {extent.width, extent.height, 1};
    copy_image(ActiveFrame().command, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer->Handle(), 1, &copy);
    VkMemoryBarrier memory{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    memory.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    memory.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    barrier(ActiveFrame().command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &memory, 0, nullptr, 0,
            nullptr);
    Transition(image, ColorAspect(), layout, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    screenshot_buffer_ = std::move(buffer);
    screenshot_width_ = extent.width;
    screenshot_height_ = extent.height;
    return true;
}

bool VulkanBackend::FinishSwapchainScreenshot() {
    if (pending_screenshot_.empty()) return true;
    if (!screenshot_buffer_) return Fail("screenshot", "post-UI screenshot buffer is missing");
    std::string error;
    if (!DrainFrameContexts()) return false;
    std::vector<uint8_t> pixels(size_t(screenshot_width_) * screenshot_height_ * 4);
    if (!screenshot_buffer_->Read(0, pixels.data(), pixels.size(), error)) return Fail("screenshot map", error);
    const VkFormat format = swapchain_.Format();
    if (format == VK_FORMAT_R8G8B8A8_UNORM || format == VK_FORMAT_R8G8B8A8_SRGB)
        for (size_t i = 0; i + 3 < pixels.size(); i += 4) std::swap(pixels[i], pixels[i + 2]);
    if (!SaveBmp(pending_screenshot_, screenshot_width_, screenshot_height_, pixels.data(), screenshot_width_ * 4))
        return Fail("screenshot save", "could not write post-UI screenshot");
    LOGI("vulkan: post-UI screenshot saved (%ux%u)", screenshot_width_, screenshot_height_);
    pending_screenshot_.clear();
    screenshot_buffer_.reset();
    return true;
}

bool VulkanBackend::RecordBlit(Target* source, const vulkan::AcquireResult& acquired) {
    EndRendering();
    VkImage swap_image = swapchain_.Image(acquired);
    VkImageLayout& swap_layout = external_layouts_[swap_image];
    if (source) Transition(*source, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    TransitionSwapchainColorAttachment(swap_image, swap_layout);
    if (!BeginRendering(nullptr, nullptr, swapchain_.View(acquired), swapchain_.Format(), swapchain_.Extent())) return false;
    const VkDeviceSize alignment = std::max<VkDeviceSize>(kConstantAlignment, context_.Properties().limits.minUniformBufferOffsetAlignment);
    const VkDeviceSize draw_offset = Align(kConstantRange, alignment);
    auto constants = std::make_unique<vulkan::Buffer>();
    std::string error;
    if (!constants->Init(context_, draw_offset + kConstantRange, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, error))
        return Fail("blit constants", error);
    UpdateGammaRamp();
    vulkan::BlitConstants data{{0, 0, 1, 1}, {1, gamma_effective_ ? 1.0f : 0.0f, 0, 0}, {}};
    for (uint32_t channel = 0; channel < 3; ++channel)
        for (uint32_t index = 0; index < 256; ++index) {
            const uint8_t value = gamma_effective_ ? gamma_lut_[channel * 256 + index] : uint8_t(index);
            data.gamma[channel * 64 + index / 4][index & 3] = float(value) / 255.0f;
        }
    if (!constants->Write(draw_offset, &data, sizeof(data), error)) return Fail("blit constant upload", error);
    vulkan::Buffer* constants_ptr = constants.get();
    ActiveFrame().frame_buffers.push_back(std::move(constants));
    std::array<VkImageView, vulkan::kTextureBindingCount> views{};
    views.fill(dummy_->View());
    if (source) views[0] = source->image->View();
    std::array<VkSampler, vulkan::kSamplerBindingCount> samplers{};
    samplers.fill(linear_sampler_);
    std::array<uint32_t, 8> offsets{};
    offsets[1] = static_cast<uint32_t>(draw_offset);
    vulkan::DescriptorSet set;
    if (!AllocateDescriptors(views, samplers, *constants_ptr, kConstantRange, set)) return false;
    auto bind_pipeline = DeviceProc<PFN_vkCmdBindPipeline>(context_, "vkCmdBindPipeline");
    auto set_viewport = DeviceProc<PFN_vkCmdSetViewport>(context_, "vkCmdSetViewport");
    auto set_scissor = DeviceProc<PFN_vkCmdSetScissor>(context_, "vkCmdSetScissor");
    auto draw = DeviceProc<PFN_vkCmdDraw>(context_, "vkCmdDraw");
    if (!bind_pipeline || !set_viewport || !set_scissor || !draw) return Fail("blit commands", "entry point unavailable");
    bind_pipeline(ActiveFrame().command, VK_PIPELINE_BIND_POINT_GRAPHICS, blit_pipeline_);
    if (!ActiveFrame().descriptors.Bind(ActiveFrame().command, pipeline_layout_, set, offsets, last_error_)) return Fail("blit descriptor bind", last_error_);
    VkViewport viewport{0, float(swapchain_.Extent().height), float(swapchain_.Extent().width), -float(swapchain_.Extent().height), 0, 1};
    VkRect2D scissor{{0, 0}, swapchain_.Extent()};
    set_viewport(ActiveFrame().command, 0, 1, &viewport);
    set_scissor(ActiveFrame().command, 0, 1, &scissor);
    draw(ActiveFrame().command, 3, 1, 0, 0);
    if (!RenderUi(acquired)) return false;
    EndRendering();
    if (!pending_screenshot_.empty() && !RecordSwapchainScreenshot(swap_image, swap_layout)) return false;
    Transition(swap_image, ColorAspect(), swap_layout, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    return true;
}

void VulkanBackend::Clear(const State& state, uint32_t count, const uint8_t* rects, uint32_t flags, uint32_t color,
                          float z, uint32_t stencil) {
    if (!BeginFrame()) return;
    Target* target = nullptr;
    Target* depth = nullptr;
    if (flags & kClearTarget) {
        if (!EnsureTarget(state.rt, false, target)) return;
    }
    if (flags & (kClearZ | kClearStencil)) {
        if (!EnsureTarget(state.zs, true, depth)) return;
    }
    if (!target && !depth) {
        Fail("clear", "clear requested without a render target or depth surface");
        return;
    }
    if (!BeginRendering(target, depth)) return;
    std::vector<VkClearRect> clear_rects;
    Target& rect_target = target ? *target : *depth;
    Rects(state, rect_target, count, rects, clear_rects);
    if (clear_rects.empty()) {
        EndRendering();
        return;
    }
    if (depth && (flags & kClearZ) && depth == sky_scene_depth_) sky_scene_done_ = true;
    std::array<VkClearAttachment, 2> attachments{};
    uint32_t attachment_count = 0;
    if (target && (flags & kClearTarget)) {
        attachments[attachment_count].aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        Color(color, attachments[attachment_count].clearValue.color.float32);
        ++attachment_count;
        target->drawn = true;
        target->drawn_generation = frame_generation_;
    }
    if (depth && (flags & (kClearZ | kClearStencil))) {
        attachments[attachment_count].aspectMask = ((flags & kClearZ) ? VK_IMAGE_ASPECT_DEPTH_BIT : 0) |
                                                    ((flags & kClearStencil) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
        attachments[attachment_count].clearValue.depthStencil = {std::clamp(z, 0.0f, 1.0f), stencil & 0xff};
        ++attachment_count;
        depth->drawn = true;
        depth->drawn_generation = frame_generation_;
    }
    auto clear = DeviceProc<PFN_vkCmdClearAttachments>(context_, "vkCmdClearAttachments");
    if (!clear) {
        Fail("clear command", "vkCmdClearAttachments unavailable");
        return;
    }
    clear(ActiveFrame().command, attachment_count, attachments.data(), static_cast<uint32_t>(clear_rects.size()), clear_rects.data());
}

void VulkanBackend::Draw(const State& state, uint32_t prim, uint32_t start, uint32_t count, const uint8_t* indices) {
    if (!BeginFrame()) return;
    Assembled assembled;
    std::chrono::steady_clock::time_point detail_start;
    if (ActiveFrame().perf_detail_frame_active) detail_start = std::chrono::steady_clock::now();
    if (!Assemble(state, prim, start, count, indices, assembled)) return;
    if (ActiveFrame().perf_detail_frame_active)
        ActiveFrame().perf_current_detail_vertex_assembly_ns += ElapsedNanoseconds(detail_start);

    if (ActiveFrame().perf_detail_frame_active) detail_start = std::chrono::steady_clock::now();
    Target* color = nullptr;
    Target* depth = nullptr;
    if (!EnsureTarget(state.rt, false, color)) return;
    if (state.rs[kRsZEnable] || state.rs[kRsStencilEnable]) {
        if (!EnsureTarget(state.zs, true, depth)) return;
    }
    if (!color && !depth) {
        Fail("draw", "draw has no render target or depth surface");
        return;
    }
    std::array<VkImageView, 4> views{};
    std::array<VkSampler, 4> samplers{};
    std::array<TexInfo, 4> infos{};
    std::array<uint64_t, 4> texture_ids{};
    Target* feedback_target = nullptr;
    VkImageView feedback_view = VK_NULL_HANDLE;
    for (int stage = 0; stage < 4; ++stage) {
        TextureDesc desc;
        Texture* texture = nullptr;
        if (!EnsureTexture(state, stage, texture, desc)) return;
        if (!state.textures[size_t(stage)]) continue;
        VkFormat sampled_format = VK_FORMAT_UNDEFINED;
        infos[size_t(stage)].kind = desc.cube ? kTexCube : desc.depth > 1 ? kTex3D : kTex2D;
        infos[size_t(stage)].rect = desc.linear ? 1 : 0;
        const FormatInfo& format = Format(desc.format);
        infos[size_t(stage)].shadow = format.depth ? vulkan::ShadowDepthKind(desc.format) : 0;
        texture_ids[size_t(stage)] = texture ? texture->id : 0;
        auto override = override_textures_.find(texture_ids[size_t(stage)]);
        if (override != override_textures_.end() && override->second.image) {
            views[size_t(stage)] = override->second.image->SampledView();
            sampled_format = override->second.image->Format();
        } else if (auto it = targets_.find(SurfaceKey(Surface{.data = desc.data}, format.depth)); it != targets_.end() &&
                   CompatibleTarget(it->second, desc, format.depth)) {
            Target& sampled_target = it->second;
            if (!PrepareTargetForSampling(sampled_target)) return;
            if (&sampled_target == color || &sampled_target == depth) {
                if (feedback_target == &sampled_target) {
                    views[size_t(stage)] = feedback_view;
                } else if (!SnapshotTargetForSampling(sampled_target, feedback_view)) {
                    return;
                } else {
                    feedback_target = &sampled_target;
                    views[size_t(stage)] = feedback_view;
                }
            } else {
                // The previous draw may still have this image inside an open dynamic-rendering scope. Vulkan image
                // barriers are not legal inside that scope, even when the next draw uses a different target.
                EndRendering();
                Transition(sampled_target, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                views[size_t(stage)] = sampled_target.image->SampledView();
            }
            sampled_format = sampled_target.image->Format();
        } else if (texture) {
            views[size_t(stage)] = texture->image->SampledView();
            sampled_format = texture->image->Format();
            texture->used_generation = frame_generation_;
        }
        if (sampled_format == VK_FORMAT_UNDEFINED) {
            Fail("texture sampler", "sampled image has no Vulkan format");
            return;
        }
        samplers[size_t(stage)] = SamplerFor(state, stage, infos[size_t(stage)].shadow != 0, sampled_format);
        if (!samplers[size_t(stage)]) return;
    }
    if (ActiveFrame().perf_detail_frame_active)
        ActiveFrame().perf_current_detail_texture_target_sampler_ns += ElapsedNanoseconds(detail_start);
    PixelKey pixel_key = MakePixelKey(state, infos.data());
    pixel_key.quad = count <= 6 && !state.rs[kRsZEnable];
    const bool flat = state.layout2d || assembled.pretransformed;
    if (flat && sky_scene_color_) sky_scene_done_ = true;
    if (g_hide_hud.load(std::memory_order_relaxed) && (state.layout2d & 0xff) == 1) return;
    DrawConstants constants{};
    const bool programmable = (state.vs_handle & 1) && state.vs && !state.vs->program.empty();
    if (g_log_draws.load(std::memory_order_relaxed) > 0) {
        const bool fog_enable = state.rs[kRsFogEnable] != 0;
        const bool range_enable = state.rs[kRsRangeFogEnable] != 0;
        const uint32_t fog_mode = state.rs[kRsFogTableMode];
        // Xbox fog uses the four low mode values; keep the per-frame mask within 64 bits.
        const uint32_t variant = (fog_enable ? 1u : 0u) | ((fog_mode & 3u) << 1) |
                                 (range_enable ? (1u << 3) : 0u) |
                                 (programmable ? (1u << 4) : 0u) | (flat ? (1u << 5) : 0u);
        const uint64_t bit = uint64_t{1} << variant;
        if ((fog_trace_seen_ & bit) == 0) {
            fog_trace_seen_ |= bit;
            LOGR("vulkan: fog trace enable %u range %u mode %u prog %u flat %u start %.6g end %.6g density %.6g color %08X",
                 fog_enable ? 1u : 0u, range_enable ? 1u : 0u, fog_mode, programmable ? 1u : 0u,
                 flat ? 1u : 0u, Bits(state.rs[kRsFogStart]), Bits(state.rs[kRsFogEnd]),
                 Bits(state.rs[kRsFogDensity]), state.rs[kRsFogColor]);
        }
    }
    FixedKey fixed = programmable ? FixedKey{} : MakeFixedKey(state);
    FillConstants(state, color ? color->guest_width : depth->guest_width, color ? color->guest_height : depth->guest_height,
                  programmable, assembled.pretransformed, fixed, infos, constants);
    if (!flat && !sky_scene_done_ && color && depth && color->guest_width == guest_width_ &&
        color->guest_height == guest_height_) {
        sky_scene_color_ = color;
        sky_scene_depth_ = depth;
    }
    const bool shell_scene_eligible =
        !flat && !sky_scene_done_ && color && color == sky_scene_color_ && sky_scene_depth_ &&
        color->guest_width == guest_width_ && color->guest_height == guest_height_;
    if (shell_scene_eligible) {
        constexpr uint32_t kAlways = 0x207;
        const bool shell_sky = sky_phase_ && state.rs[kRsZEnable] && SimpleState(state, kMZWriteEnable) &&
                               SimpleState(state, kMZFunc) == kAlways;
        if (g_level.load(std::memory_order_relaxed) == 1) {
            if (shell_sky && sky_layer_ < 16) {
                constants.jitter[3] = 1.0f - float(sky_layer_++) * kSkyLayerStep;
            } else {
                sky_phase_ = false;
            }
        } else if (shell_sky) {
            constants.jitter[3] = 1.0f;
        } else {
            sky_phase_ = false;
        }
    }
    recolor_ = false;
    if (state.layout2d || assembled.pretransformed) {
        Draw2D draw;
        BuildDraw2D(state, assembled, programmable, color ? color->guest_width : depth->guest_width,
                    color ? color->guest_height : depth->guest_height, texture_ids[0], draw, recolor_element_);
        draw.widen = g_screen_widen.load(std::memory_order_relaxed);
        draw.frame = frame_count_;
        if (draw.bounded) {
            Draw2DSeen seen{draw.x0, draw.y0, draw.x1, draw.y1, draw.tex, draw.color, draw.tag, draw.frame};
            if (auto callback = g_seen_2d.load(std::memory_order_relaxed)) callback(seen);
            uint32_t recolor_argb = 0;
            if (auto callback = g_recolor_2d.load(std::memory_order_relaxed);
                callback && (assembled.regs & 8) && recolor_element_.format == DXGI_FORMAT_B8G8R8A8_UNORM &&
                    callback(seen, recolor_argb)) {
                recolor_ = true;
                recolor_argb_ = recolor_argb;
            }
        }
        auto layout_callback = g_layout_2d.load(std::memory_order_relaxed);
        const Target* screen_target = color ? color : depth;
        if (layout_callback && screen_target && screen_target->guest_width == guest_width_ &&
            screen_target->guest_height == guest_height_) {
            Place2D place;
            if (layout_callback(draw, place)) {
                constants.layout[0] = place.sx;
                constants.layout[1] = place.ox;
                constants.layout_y[0] = place.sy;
                constants.layout_y[1] = place.oy;
            }
        }
    }
    if (!RecordDraw(state, assembled, pixel_key, color, depth, views, samplers, constants)) return;
}

void VulkanBackend::Copy(const Surface& src, const Surface& dst, uint32_t count, uint32_t src_rects,
                         uint32_t dst_points) {
    TextureDesc source_desc;
    if (DescribeSurface(src, source_desc) && Format(source_desc.format).depth) {
        if (!BeginFrame()) return;
        EndRendering();
        if (!CopySurfaceMemory(src, dst, count, src_rects, dst_points)) {
            Fail("copy", "depth-source CPU surface copy does not support the requested formats");
            return;
        }
        TextureDesc destination_desc;
        if (!DescribeSurface(dst, destination_desc) || Format(destination_desc.format).depth) return;
        auto destination_it = targets_.find(SurfaceKey(dst, false));
        if (destination_it == targets_.end()) return;
        Target& destination = destination_it->second;
        if (destination.depth || !destination.image || destination.guest_width != destination_desc.width ||
            destination.guest_height != destination_desc.height)
            return;
        destination.surface = dst;
        destination.drawn = false;
        destination.readback.clear();
        if (!UploadSurface(destination)) return;
        return;
    }
    if (!BeginFrame()) return;
    EndRendering();
    Target* source = nullptr;
    Target* destination = nullptr;
    if (!EnsureTarget(src, false, source) || !EnsureTarget(dst, false, destination)) return;
    if (source && destination && source == destination && source->drawn) {
        // Vulkan cannot legally use one image as both overlapping copy source and destination. Resolve the current
        // target to guest memory first, then use CopySurfaceMemory's source snapshot for the requested move.
        if (!ReadBack(*source, source->surface.data) || !BeginFrame()) return;
    }
    if (!source || !source->drawn) {
        if (!CopySurfaceMemory(src, dst, count, src_rects, dst_points)) {
            Fail("copy", "CPU surface copy does not support the requested formats");
            return;
        }
        if (destination) {
            destination->drawn = false;
            destination->readback.clear();
            if (!UploadSurface(*destination)) return;
        }
        return;
    }
    if (!destination) {
        Fail("copy", "GPU surface copy destination is unavailable");
        return;
    }
    RecordTargetCopy(*source, *destination, count, src_rects, dst_points);
}

bool VulkanBackend::ReadBackPixels(Target& target, std::vector<uint8_t>& pixels) {
    if (target.depth || !target.image) return false;
    TextureDesc surface_desc;
    if (!DescribeSurface(target.surface, surface_desc))
        return Fail("readback description", "render target surface description is invalid");
    if (!CanWriteBackSurface(surface_desc))
        return Fail("readback format", "guest surface storage is not a writable linear 32-bit color format");
    EndRendering();
    Transition(target, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    const VkDeviceSize bytes = VkDeviceSize(target.width) * target.height * 4;
    auto readback = std::make_unique<vulkan::Buffer>();
    std::string error;
    if (!readback->Init(context_, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, error))
        return Fail("readback buffer", error);
    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = {target.width, target.height, 1};
    auto copy_image = DeviceProc<PFN_vkCmdCopyImageToBuffer>(context_, "vkCmdCopyImageToBuffer");
    auto barrier = DeviceProc<PFN_vkCmdPipelineBarrier>(context_, "vkCmdPipelineBarrier");
    if (!copy_image || !barrier) return Fail("readback command", "required Vulkan readback entry point unavailable");
    copy_image(ActiveFrame().command, target.image->Handle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback->Handle(), 1, &copy);
    VkMemoryBarrier memory{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    memory.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    memory.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    barrier(ActiveFrame().command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &memory, 0, nullptr, 0, nullptr);
    Transition(target, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    if (!CompleteFrame() || !DrainFrameContexts()) return false;
    pixels.resize(static_cast<size_t>(bytes));
    if (!readback->Read(0, pixels.data(), pixels.size(), error)) return Fail("readback map", error);
    return true;
}

bool VulkanBackend::ReadBack(Target& target, uint32_t data_va) {
    if (target.depth || !target.drawn) return true;
    if (!BeginFrame()) return false;
    TextureDesc surface_desc;
    if (!DescribeSurface(target.surface, surface_desc))
        return Fail("readback description", "render target surface description is invalid");
    if (!CanWriteBackSurface(surface_desc) || !surface_desc.width || !surface_desc.height ||
        surface_desc.pitch < uint64_t(surface_desc.width) * 4)
        return Fail("readback format", "guest surface storage is not a writable linear 32-bit color format");
    std::vector<uint8_t> pixels;
    if (!ReadBackPixels(target, pixels)) return false;
    uint8_t* guest = static_cast<uint8_t*>(kernel::HostPtr(data_va));
    const size_t guest_bytes = size_t(surface_desc.pitch) * surface_desc.height;
    if (guest_bytes / surface_desc.height != surface_desc.pitch ||
        !WriteBackSurface(surface_desc, pixels.data(), target.width, target.height, target.width * 4, pixels.size(), guest,
                          guest_bytes, &target.readback))
        return Fail("readback write", "guest surface writeback rejected by the shared format, pitch or bounds contract");
    if (perf_counters_enabled_.load(std::memory_order_relaxed))
        perf_window_guest_writeback_successes_.fetch_add(1, std::memory_order_relaxed);
    target.content_epoch = DirtyEpoch(surface_desc.data);
    target.content_signature = ContentSignature(surface_desc);
    target.content_generation = frame_generation_;
    target.drawn = false;
    return true;
}

void VulkanBackend::Sync(uint32_t data_va, uint32_t size) {
    if (perf_counters_enabled_.load(std::memory_order_relaxed))
        perf_window_sync_requests_.fetch_add(1, std::memory_order_relaxed);
    const uint64_t request_begin = data_va;
    const uint64_t request_end = request_begin + (size ? size : 1);
    for (auto& [key, target] : targets_)
        if (!target.depth && target.drawn) {
            TextureDesc desc;
            if (!DescribeSurface(target.surface, desc)) {
                Fail("sync description", "render target surface description is invalid");
                return;
            }
            const uint64_t begin = target.surface.data;
            const uint64_t end = begin + uint64_t(desc.pitch ? desc.pitch * desc.height : target.guest_width * target.guest_height * 4);
            if (request_begin < end && begin < request_end) {
                if (!ReadBack(target, target.surface.data)) return;
            }
        }
}

void VulkanBackend::StampLateSubmissions() {
    const uint64_t serial = ActiveFrame().submission_ordinal;
    late_published_generation_.store(frame_generation_, std::memory_order_release);
    if (!serial) return;
    std::lock_guard lock(late_lock_);
    for (auto& [data, late] : late_)
        for (auto& slot : late.states)
            if (slot.valid && !slot.submission_serial && slot.frame_generation == frame_generation_)
                slot.submission_serial = serial;
}

void VulkanBackend::PublishLateCompletion() {
    const int prior_error = late_published_error_.load(std::memory_order_acquire);
    if (prior_error != VK_SUCCESS) {
        Fail("late readback", "a previous late-copy operation failed");
        return;
    }
    uint64_t completed = completed_submission_ordinal_;
    for (const auto& frame : frame_contexts_) {
        if (!frame.batch.Pending()) continue;
        const VkResult status = frame.batch.CompletionStatus();
        if (status == VK_SUCCESS) {
            completed = std::max(completed, frame.submission_ordinal);
            continue;
        }
        if (status != VK_NOT_READY) {
            late_published_error_.store(static_cast<int>(status), std::memory_order_release);
            Fail("late completion", "Vulkan fence status reported an unexpected error");
            return;
        }
    }
    completed_submission_ordinal_ = completed;
    late_published_completed_serial_.store(completed, std::memory_order_release);
}

void VulkanBackend::SyncLate(uint32_t data_va) {
    if (!DeliverLate(data_va)) Sync(data_va, 0);
    RecordLate(data_va);
}

bool VulkanBackend::DeliverLate(uint32_t data_va) {
    const auto miss = [this]() {
        if (perf_counters_enabled_.load(std::memory_order_relaxed))
            perf_window_deliver_late_misses_.fetch_add(1, std::memory_order_relaxed);
        return false;
    };
    const auto hit = [this]() {
        if (perf_counters_enabled_.load(std::memory_order_relaxed))
            perf_window_deliver_late_hits_.fetch_add(1, std::memory_order_relaxed);
        return true;
    };
    if (late_published_error_.load(std::memory_order_acquire) != VK_SUCCESS) return miss();
    std::lock_guard lock(late_lock_);
    auto it = late_.find(data_va);
    if (it == late_.end() || !it->second.configured) return miss();
    Late& late = it->second;
    const uint64_t completed = late_published_completed_serial_.load(std::memory_order_acquire);
    const uint64_t published = late_published_generation_.load(std::memory_order_acquire);
    const int best = vulkan::LateDeliverySlot(late.states, completed, published, late.identity_generation,
                                               Late::kSlotCount + 1);
    if (best < 0) return miss();
    LateSlot& slot = late.slots[best];
    const auto& state = late.states[best];
    if (!slot.readback || slot.desc.data != data_va || !SameLateDesc(slot.desc, late.desc) ||
        slot.width != late.width || slot.height != late.height || slot.image_identity != late.image_identity)
        return miss();
    if (!vulkan::LateNeedsWrite(state, late.delivered_serial)) return hit();
    const size_t guest_bytes = size_t(slot.desc.pitch) * slot.desc.height;
    if (!slot.desc.width || !slot.desc.height || guest_bytes / slot.desc.height != slot.desc.pitch) return miss();
    std::vector<uint8_t> pixels(size_t(slot.width) * slot.height * 4);
    std::string error;
    if (!slot.readback->Read(0, pixels.data(), pixels.size(), error)) {
        late_published_error_.store(VK_ERROR_UNKNOWN, std::memory_order_release);
        return miss();
    }
    uint8_t* guest = static_cast<uint8_t*>(kernel::HostPtr(data_va));
    if (!WriteBackSurface(slot.desc, pixels.data(), slot.width, slot.height, slot.width * 4, pixels.size(), guest,
                          guest_bytes, nullptr))
        return miss();
    if (perf_counters_enabled_.load(std::memory_order_relaxed))
        perf_window_guest_writeback_successes_.fetch_add(1, std::memory_order_relaxed);
    late.delivered_serial = state.submission_serial;
    late.states[best].delivered_serial = state.submission_serial;
    late.acknowledged_epoch = DirtyEpoch(slot.desc.data);
    late.acknowledged_signature = ContentSignature(slot.desc);
    late.acknowledged_identity = late.identity_generation;
    late.acknowledgement_pending = true;
    return hit();
}

void VulkanBackend::RecordLate(uint32_t data_va) {
    const auto skip = [this]() {
        if (perf_counters_enabled_.load(std::memory_order_relaxed))
            perf_window_record_late_skips_.fetch_add(1, std::memory_order_relaxed);
    };
    auto target_it = targets_.find(SurfaceKey(Surface{.data = data_va}, false));
    if (target_it == targets_.end() || target_it->second.depth || !target_it->second.image) {
        skip();
        return;
    }
    Target& target = target_it->second;
    TextureDesc desc;
    if (!DescribeSurface(target.surface, desc) || desc.data != data_va || !CanWriteBackSurface(desc) || !desc.width ||
        !desc.height || desc.pitch < uint64_t(desc.width) * 4) {
        skip();
        return;
    }
    if (!BeginFrame()) {
        skip();
        return;
    }
    const VkDeviceSize bytes = VkDeviceSize(target.width) * target.height * 4;
    if (!bytes || bytes / target.height / 4 != target.width) {
        skip();
        return;
    }
    std::lock_guard lock(late_lock_);
    Late& late = late_[data_va];
    const uint64_t completed = completed_submission_ordinal_;
    const bool shape_changed = late.configured &&
                               (!SameLateDesc(late.desc, desc) || late.width != target.width || late.height != target.height ||
                                late.image_identity != target.image->Handle());
    if (shape_changed) {
        late.desc = desc;
        late.width = target.width;
        late.height = target.height;
        late.image_identity = target.image->Handle();
        ++late.identity_generation;
        late.retire_pending = true;
        late.acknowledgement_pending = false;
    }
    if (late.retire_pending) {
        bool all_retired = true;
        all_retired = vulkan::LateSlotsRetired(late.states, completed);
        if (!all_retired) {
            skip();
            return;
        }
        late.slots = {};
        late.states = {};
        late.next = 0;
        late.delivered_serial = 0;
        late.retire_pending = false;
    }
    if (!late.configured) {
        late.desc = desc;
        late.width = target.width;
        late.height = target.height;
        late.image_identity = target.image->Handle();
        late.identity_generation = 1;
        late.configured = true;
    }
    const int slot_index = vulkan::LateRecordSlot(late.states, late.next, completed);
    if (slot_index < 0) {
        skip();
        return;
    }
    LateSlot& slot = late.slots[slot_index];
    if (!slot.readback || slot.readback->Size() != bytes) {
        if (!slot.readback) slot.readback = std::make_unique<vulkan::Buffer>();
        std::string error;
        if (!slot.readback->Init(context_, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, error))
            return (void)(skip(), Fail("late readback buffer", error));
    }
    EndRendering();
    Transition(target, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = {target.width, target.height, 1};
    auto copy_image = DeviceProc<PFN_vkCmdCopyImageToBuffer>(context_, "vkCmdCopyImageToBuffer");
    auto barrier2 = DeviceProc<PFN_vkCmdPipelineBarrier2>(context_, "vkCmdPipelineBarrier2");
    if (!copy_image || !barrier2) {
        Transition(target, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        return (void)(skip(), Fail("late readback command", "required Vulkan copy/barrier entry point unavailable"));
    }
    copy_image(ActiveFrame().command, target.image->Handle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, slot.readback->Handle(), 1, &copy);
    VkBufferMemoryBarrier2 host{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
    host.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    host.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    host.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    host.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
    host.srcQueueFamilyIndex = host.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    host.buffer = slot.readback->Handle();
    host.offset = 0;
    host.size = bytes;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.bufferMemoryBarrierCount = 1;
    dependency.pBufferMemoryBarriers = &host;
    barrier2(ActiveFrame().command, &dependency);
    Transition(target, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    slot.desc = desc;
    slot.width = target.width;
    slot.height = target.height;
    slot.image_identity = target.image->Handle();
    late.states[slot_index] = {frame_generation_, late.identity_generation, 0, 0, true};
    late.next = (late.next + 1) % Late::kSlotCount;
    if (perf_counters_enabled_.load(std::memory_order_relaxed))
        perf_window_record_late_accepted_.fetch_add(1, std::memory_order_relaxed);
    target.content_epoch = DirtyEpoch(desc.data);
    target.content_signature = ContentSignature(desc);
    target.content_generation = frame_generation_;
    target.readback.clear();
}

void VulkanBackend::Present(const Surface& front) {
    fog_trace_seen_ = 0;
    sky_phase_ = true;
    sky_layer_ = 0;
    sky_scene_done_ = false;
    sky_scene_color_ = sky_scene_depth_ = nullptr;
    perf_trace_.Present(g_frame_stats.load() == 1);
    if (!BeginFrame()) return;
    Target* source = nullptr;
    if (!EnsureTarget(front, false, source)) return;
    if (source && !source->drawn && LatchCpuWrite(*source) && !UploadSurface(*source)) return;
    std::wstring shot;
    if (control::TakeShot(shot)) {
        pending_screenshot_ = std::move(shot);
        screenshot_buffer_.reset();
    }
    std::string error;
    vulkan::AcquireResult acquired = swapchain_.Acquire(UINT64_MAX, error);
    if (!acquired.Succeeded()) {
        const bool recreate_requested = acquired.status == vulkan::AcquireStatus::RecreateRequired ||
                                        (acquired.status == vulkan::AcquireStatus::Suspended && !swapchain_.Ready());
        if (recreate_requested) {
            if (!DrainFrameContexts()) return;
            const VkFormat old_format = swapchain_.Format();
            const uint32_t old_image_count = swapchain_.ImageCount();
            app::Mode output = app::Output();
            const vulkan::SwapchainInitStatus recreate_status =
                swapchain_.Recreate({std::max(output.width, 1u), std::max(output.height, 1u)},
                                     app::Get().vsync ? vulkan::PresentPreference::Fifo
                                                      : vulkan::PresentPreference::Mailbox,
                                     error);
            if (recreate_status == vulkan::SwapchainInitStatus::Ready) {
                external_layouts_.clear();
                if (old_format != swapchain_.Format()) {
                    if (!RecreateUi()) {
                        CompleteFrame();
                        return;
                    }
                } else if (ui_initialized_ && old_image_count != swapchain_.ImageCount()) {
                    std::lock_guard lock(app::UiLock());
                    ImGui_ImplVulkan_SetMinImageCount(std::max(2u, swapchain_.ImageCount()));
                }
                if (!BeginFrame()) return;
                acquired = swapchain_.Acquire(UINT64_MAX, error);
            } else if (recreate_status == vulkan::SwapchainInitStatus::Suspended) {
                CompleteFrame();
                return;
            } else {
                if (error.empty()) {
                    switch (recreate_status) {
                        case vulkan::SwapchainInitStatus::CapabilityLimited:
                            error = "Vulkan swapchain recreate returned CapabilityLimited";
                            break;
                        case vulkan::SwapchainInitStatus::SurfaceLost:
                            error = "Vulkan swapchain recreate returned SurfaceLost";
                            break;
                        case vulkan::SwapchainInitStatus::DeviceLost:
                            error = "Vulkan swapchain recreate returned DeviceLost";
                            break;
                        case vulkan::SwapchainInitStatus::Failed:
                            error = "Vulkan swapchain recreate returned Failed";
                            break;
                        case vulkan::SwapchainInitStatus::Ready:
                        case vulkan::SwapchainInitStatus::Suspended:
                            break;
                    }
                }
                CompleteFrame();
                Fail("swapchain recreate", error);
                return;
            }
        }
        if (!acquired.Succeeded()) {
            CompleteFrame();
            if (acquired.status == vulkan::AcquireStatus::Suspended || acquired.status == vulkan::AcquireStatus::NotReady ||
                acquired.status == vulkan::AcquireStatus::Timeout ||
                acquired.status == vulkan::AcquireStatus::RecreateRequired)
                return;
            Fail("swapchain acquire", error);
            return;
        }
    }
    if (!RecordBlit(source, acquired)) {
        EndRendering();
        std::string abandon;
        swapchain_.Abandon(acquired, abandon);
        return;
    }
    EndPerfFrame();
    if (!ActiveFrame().batch.EndForExternalSubmit(error)) {
        EndRendering();
        std::string abandon;
        swapchain_.Abandon(acquired, abandon);
        Fail("external command end", error);
        return;
    }
    const VkFence prior_image_fence = swapchain_.RenderFenceForImage(acquired.image_index);
    for (uint32_t i = 0; i < kFrameContextCount; ++i) {
        if (!prior_image_fence || i == active_frame_index_ || frame_contexts_[i].completion_fence != prior_image_fence ||
            !frame_contexts_[i].batch.Pending())
            continue;
        if (!RetireFrameContext(i, UINT64_MAX, "swapchain render-fence owner retirement")) {
            EndRendering();
            std::string abandon;
            swapchain_.Abandon(acquired, abandon);
            return;
        }
    }
    if (!swapchain_.Submit(acquired, ActiveFrame().command, error, vulkan::SubmitWaitPolicy::ColorAttachmentOutput)) {
        EndRendering();
        std::string abandon;
        swapchain_.Abandon(acquired, abandon);
        Fail("swapchain submit", error);
        return;
    }
    const VkFence completion_fence = swapchain_.RenderFence(acquired);
    if (!ActiveFrame().batch.MarkExternalSubmitted(completion_fence, error)) {
        std::string completion_error;
        swapchain_.WaitForRendering(UINT64_MAX, completion_error);
        ActiveFrame().batch.Wait(UINT64_MAX, completion_error);
        std::string abandon;
        swapchain_.Abandon(acquired, abandon);
        Fail("external completion adoption", error);
        return;
    }
    ActiveFrame().recording = false;
    ActiveFrame().submission_ordinal = ++next_submission_ordinal_;
    ActiveFrame().completion_fence = completion_fence;
    StampLateSubmissions();
    if (ActiveFrame().descriptors_used && !ActiveFrame().descriptors.MarkInFlight(ActiveFrame().batch, error)) {
        std::string completion_error;
        swapchain_.WaitForRendering(UINT64_MAX, completion_error);
        ActiveFrame().batch.Wait(UINT64_MAX, completion_error);
        swapchain_.Abandon(acquired, completion_error);
        Fail("present descriptor ownership", error);
        return;
    }
    vulkan::PresentStatus status = swapchain_.Present(acquired, error);
    if (status == vulkan::PresentStatus::DeviceLost || status == vulkan::PresentStatus::Failed ||
        status == vulkan::PresentStatus::SurfaceLost)
        Fail("swapchain present", error);
    if (status == vulkan::PresentStatus::Presented && !FinishSwapchainScreenshot()) return;
    if (g_log_draws > 0 && --g_log_draws == 0) LOGR("draw: end of frame");
    ++frame_count_;
    g_presented.fetch_add(1, std::memory_order_relaxed);
    const auto now = std::chrono::steady_clock::now();
    if (fps_sample_time_ == std::chrono::steady_clock::time_point{}) fps_sample_time_ = now;
    const double elapsed = std::chrono::duration<double>(now - fps_sample_time_).count();
    if (elapsed >= 0.5) {
        fps_ = double(frame_count_ - fps_sample_frame_) / elapsed;
        fps_sample_frame_ = frame_count_;
        fps_sample_time_ = now;
    }
}

void VulkanBackend::Flush() {
    if (CompleteFrame()) DrainFrameContexts();
}

void VulkanBackend::PrepareShutdown() {
    if (!context_.Initialized()) return;
    Flush();
    if (perf_enabled_) ReportPerfWindow();
    if (!context_.PersistPipelineCache())
        LOGW("vulkan: persistent pipeline cache was not written during shutdown");
}

std::unique_ptr<Backend> CreateVulkanBackend() { return std::make_unique<VulkanBackend>(); }

}  // namespace gpu
