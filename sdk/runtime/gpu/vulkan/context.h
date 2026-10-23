#pragma once

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include "gpu/vulkan/loader.h"
#include "gpu/vulkan/pipeline_cache.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>

namespace gpu::vulkan {

class Context {
public:
    Context() = default;
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    ~Context();

    // Creates a capability-checked Vulkan device without creating a window surface.
    bool Init(std::string& error, bool enable_win32_wsi = false);
    void Reset();

    bool Initialized() const { return device_ != VK_NULL_HANDLE; }
    VkInstance Instance() const { return instance_; }
    VkPhysicalDevice PhysicalDevice() const { return physical_device_; }
    VkDevice Device() const { return device_; }
    VkQueue GraphicsQueue() const { return graphics_queue_; }
    uint32_t GraphicsQueueFamily() const { return graphics_queue_family_; }
    const VkPhysicalDeviceProperties& Properties() const { return properties_; }
    const std::array<uint8_t, VK_UUID_SIZE>& DeviceUUID() const { return device_uuid_; }
    void SetPipelineCacheDirectory(std::filesystem::path directory) { pipeline_cache_directory_ = std::move(directory); }
    VkPipelineCache PipelineCacheHandle() const { return pipeline_cache_.Handle(); }
    const PipelineCacheStats& PipelineCacheStatus() const { return pipeline_cache_.Stats(); }
    bool PersistPipelineCache();
    bool SwapchainMaintenance1Enabled() const { return swapchain_maintenance1_enabled_; }
    bool SamplerMirrorClampToEdgeEnabled() const { return sampler_mirror_clamp_to_edge_enabled_; }
    bool CustomBorderColorExtensionAvailable() const { return custom_border_color_extension_available_; }
    bool CustomBorderColorFeatureAvailable() const { return custom_border_color_features_available_; }
    bool CustomBorderColorEnabled() const { return custom_border_color_enabled_; }
    bool CustomBorderColorWithoutFormatEnabled() const { return custom_border_color_without_format_enabled_; }
    uint32_t MaxCustomBorderColorSamplers() const { return max_custom_border_color_samplers_; }
    bool CustomBorderColorFormatRequiresTyped(VkFormat format) const;
    bool CustomBorderColorFormatSupported(VkFormat format) const;
    bool ValidationAvailable() const { return validation_available_; }
    bool ValidationEnabled() const { return validation_enabled_; }
    uint32_t DebugWarningCount() const { return debug_warning_count_.load(std::memory_order_relaxed); }
    uint32_t DebugErrorCount() const { return debug_error_count_.load(std::memory_order_relaxed); }
    uint32_t DebugLogCount() const { return debug_log_count_.load(std::memory_order_relaxed); }

    PFN_vkVoidFunction InstanceProc(const char* name) const;
    PFN_vkVoidFunction DeviceProc(const char* name) const;

private:
    bool CreateInstance(std::string& error);
    bool SelectPhysicalDevice(std::string& error);
    bool CreateDevice(std::string& error);
    bool CheckPhysicalDevice(VkPhysicalDevice physical, VkPhysicalDeviceProperties& properties,
                             uint32_t& queue_family, std::string& error) const;
    bool CheckFormats(VkPhysicalDevice physical, std::string& error) const;
    bool HasDeviceExtension(VkPhysicalDevice physical, const char* name) const;

    static VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                         VkDebugUtilsMessageTypeFlagsEXT type,
                                                         const VkDebugUtilsMessengerCallbackDataEXT* data,
                                                         void* user_data);

    Loader loader_;
    VkInstance instance_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties_{};
    std::array<uint8_t, VK_UUID_SIZE> device_uuid_{};
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue graphics_queue_ = VK_NULL_HANDLE;
    uint32_t graphics_queue_family_ = VK_QUEUE_FAMILY_IGNORED;

    PFN_vkDestroyInstance destroy_instance_ = nullptr;
    PFN_vkEnumeratePhysicalDevices enumerate_physical_devices_ = nullptr;
    PFN_vkGetPhysicalDeviceProperties get_physical_device_properties_ = nullptr;
    PFN_vkGetPhysicalDeviceProperties2 get_physical_device_properties2_ = nullptr;
    PFN_vkGetPhysicalDeviceFeatures2 get_physical_device_features2_ = nullptr;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties get_queue_family_properties_ = nullptr;
    PFN_vkEnumerateDeviceExtensionProperties enumerate_device_extensions_ = nullptr;
    PFN_vkGetPhysicalDeviceFormatProperties get_format_properties_ = nullptr;
    PFN_vkGetPhysicalDeviceImageFormatProperties get_image_format_properties_ = nullptr;
    PFN_vkCreateDevice create_device_ = nullptr;
    PFN_vkGetDeviceProcAddr get_device_proc_addr_ = nullptr;
    PFN_vkDestroyDevice destroy_device_ = nullptr;
    PFN_vkDeviceWaitIdle device_wait_idle_ = nullptr;
    PFN_vkGetDeviceQueue get_device_queue_ = nullptr;
    PipelineCache pipeline_cache_;
    std::filesystem::path pipeline_cache_directory_;
    PFN_vkCreateDebugUtilsMessengerEXT create_debug_messenger_ = nullptr;
    PFN_vkDestroyDebugUtilsMessengerEXT destroy_debug_messenger_ = nullptr;
    bool enable_win32_wsi_ = false;
    bool surface_maintenance1_enabled_ = false;
    bool get_surface_capabilities2_enabled_ = false;
    bool swapchain_maintenance1_enabled_ = false;
    bool sampler_mirror_clamp_to_edge_enabled_ = false;
    bool custom_border_color_extension_available_ = false;
    bool custom_border_color_features_available_ = false;
    bool custom_border_color_without_format_available_ = false;
    bool custom_border_color_enabled_ = false;
    bool custom_border_color_without_format_enabled_ = false;
    uint32_t max_custom_border_color_samplers_ = 0;
    bool validation_available_ = false;
    bool validation_enabled_ = false;
    std::atomic<uint32_t> debug_warning_count_{0};
    std::atomic<uint32_t> debug_error_count_{0};
    std::atomic<uint32_t> debug_log_count_{0};
};

}  // namespace gpu::vulkan
