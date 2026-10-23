#include "gpu/vulkan/context.h"

#include "core/log.h"
#include "gpu/shaders.h"
#include "gpu/vulkan/shader.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

namespace gpu::vulkan {
namespace {

constexpr uint32_t kRequiredConstantBuffers = kConstantBindingCount;
constexpr uint32_t kRequiredSampledImages = kTextureBindingCount;
constexpr uint32_t kRequiredSamplers = kSamplerBindingCount;
constexpr uint32_t kRequiredDescriptorsPerStage =
    kRequiredConstantBuffers + kRequiredSampledImages + kRequiredSamplers;
constexpr uint32_t kRequiredUniformBufferRange =
    static_cast<uint32_t>(std::max<VkDeviceSize>(sizeof(::gpu::DrawConstants), kVertexRegisterBytes));
constexpr uint32_t kMaxDebugLogMessages = 64;
constexpr char kWin32SurfaceExtensionName[] = "VK_KHR_win32_surface";

bool HasExtension(const std::vector<VkExtensionProperties>& extensions, const char* name) {
    return std::any_of(extensions.begin(), extensions.end(), [name](const VkExtensionProperties& extension) {
        return std::strcmp(extension.extensionName, name) == 0;
    });
}

bool HasLayer(const std::vector<VkLayerProperties>& layers, const char* name) {
    return std::any_of(layers.begin(), layers.end(), [name](const VkLayerProperties& layer) {
        return std::strcmp(layer.layerName, name) == 0;
    });
}

std::string ResultText(VkResult result) {
    return "VkResult " + std::to_string(static_cast<int>(result));
}

bool AtLeast13(uint32_t version) {
    return VK_API_VERSION_MAJOR(version) > 1 ||
           (VK_API_VERSION_MAJOR(version) == 1 && VK_API_VERSION_MINOR(version) >= 3);
}

template <typename T>
T InstanceFunction(const Loader& loader, VkInstance instance, const char* name) {
    if (!loader.GetInstanceProcAddr()) return nullptr;
    return reinterpret_cast<T>(loader.GetInstanceProcAddr()(instance, name));
}

struct ImageRequirement {
    VkFormat format;
    const char* name;
    VkFormatFeatureFlags features;
    VkImageUsageFlags usage;
};

}  // namespace

Context::~Context() {
    Reset();
}

bool Context::Init(std::string& error, bool enable_win32_wsi) {
    Reset();
    enable_win32_wsi_ = enable_win32_wsi;
    if (!loader_.Load(error)) return false;
    if (!CreateInstance(error)) {
        Reset();
        return false;
    }
    if (!SelectPhysicalDevice(error)) {
        Reset();
        return false;
    }
    if (!CreateDevice(error)) {
        Reset();
        return false;
    }
    return true;
}

void Context::Reset() {
    if (device_ != VK_NULL_HANDLE) {
        const bool idle = device_wait_idle_ && device_wait_idle_(device_) == VK_SUCCESS;
        pipeline_cache_.Reset(idle);
        if (destroy_device_) destroy_device_(device_, nullptr);
    } else {
        pipeline_cache_.Reset();
    }
    device_ = VK_NULL_HANDLE;
    graphics_queue_ = VK_NULL_HANDLE;
    graphics_queue_family_ = VK_QUEUE_FAMILY_IGNORED;
    destroy_device_ = nullptr;
    device_wait_idle_ = nullptr;
    get_device_queue_ = nullptr;
    get_device_proc_addr_ = nullptr;

    if (debug_messenger_ != VK_NULL_HANDLE && destroy_debug_messenger_) {
        destroy_debug_messenger_(instance_, debug_messenger_, nullptr);
    }
    debug_messenger_ = VK_NULL_HANDLE;
    create_debug_messenger_ = nullptr;
    destroy_debug_messenger_ = nullptr;

    if (instance_ != VK_NULL_HANDLE && destroy_instance_) destroy_instance_(instance_, nullptr);
    instance_ = VK_NULL_HANDLE;
    physical_device_ = VK_NULL_HANDLE;
    properties_ = {};
    device_uuid_.fill(0);
    destroy_instance_ = nullptr;
    enumerate_physical_devices_ = nullptr;
    get_physical_device_properties_ = nullptr;
    get_physical_device_properties2_ = nullptr;
    get_physical_device_features2_ = nullptr;
    get_queue_family_properties_ = nullptr;
    enumerate_device_extensions_ = nullptr;
    get_format_properties_ = nullptr;
    get_image_format_properties_ = nullptr;
    create_device_ = nullptr;
    enable_win32_wsi_ = false;
    surface_maintenance1_enabled_ = false;
    get_surface_capabilities2_enabled_ = false;
    swapchain_maintenance1_enabled_ = false;
    sampler_mirror_clamp_to_edge_enabled_ = false;
    custom_border_color_extension_available_ = false;
    custom_border_color_features_available_ = false;
    custom_border_color_without_format_available_ = false;
    custom_border_color_enabled_ = false;
    custom_border_color_without_format_enabled_ = false;
    max_custom_border_color_samplers_ = 0;
    validation_available_ = false;
    validation_enabled_ = false;
    debug_warning_count_.store(0, std::memory_order_relaxed);
    debug_error_count_.store(0, std::memory_order_relaxed);
    debug_log_count_.store(0, std::memory_order_relaxed);
    loader_.Reset();
}

bool Context::PersistPipelineCache() {
    if (device_ == VK_NULL_HANDLE || !device_wait_idle_ || device_wait_idle_(device_) != VK_SUCCESS) return false;
    return pipeline_cache_.Persist();
}

PFN_vkVoidFunction Context::InstanceProc(const char* name) const {
    if (instance_ == VK_NULL_HANDLE || !loader_.GetInstanceProcAddr()) return nullptr;
    return loader_.GetInstanceProcAddr()(instance_, name);
}

PFN_vkVoidFunction Context::DeviceProc(const char* name) const {
    if (device_ == VK_NULL_HANDLE || !get_device_proc_addr_) return nullptr;
    return get_device_proc_addr_(device_, name);
}

bool Context::CustomBorderColorFormatSupported(VkFormat format) const {
    if (!custom_border_color_enabled_ || physical_device_ == VK_NULL_HANDLE || !get_format_properties_ ||
        format == VK_FORMAT_UNDEFINED)
        return false;
    VkFormatProperties properties{};
    get_format_properties_(physical_device_, format, &properties);
    return (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0;
}

bool Context::CustomBorderColorFormatRequiresTyped(VkFormat format) const {
    switch (format) {
        case VK_FORMAT_B4G4R4A4_UNORM_PACK16:
        case VK_FORMAT_B5G6R5_UNORM_PACK16:
        case VK_FORMAT_B5G5R5A1_UNORM_PACK16:
        case VK_FORMAT_A1B5G5R5_UNORM_PACK16_KHR:
            return true;
        default:
            return false;
    }
}

bool Context::CreateInstance(std::string& error) {
    auto enumerate_version = loader_.Global<PFN_vkEnumerateInstanceVersion>("vkEnumerateInstanceVersion");
    uint32_t instance_version = VK_API_VERSION_1_0;
    if (enumerate_version) {
        const VkResult result = enumerate_version(&instance_version);
        if (result != VK_SUCCESS) {
            error = "vkEnumerateInstanceVersion failed: " + ResultText(result);
            return false;
        }
    }
    if (!AtLeast13(instance_version)) {
        error = "Vulkan loader exposes less than API 1.3";
        return false;
    }

    auto enumerate_extensions = loader_.Global<PFN_vkEnumerateInstanceExtensionProperties>(
        "vkEnumerateInstanceExtensionProperties");
    auto enumerate_layers = loader_.Global<PFN_vkEnumerateInstanceLayerProperties>(
        "vkEnumerateInstanceLayerProperties");
    auto create_instance = loader_.Global<PFN_vkCreateInstance>("vkCreateInstance");
    destroy_instance_ = loader_.Export<PFN_vkDestroyInstance>("vkDestroyInstance");
    if (!enumerate_extensions || !create_instance || !destroy_instance_) {
        error = "Vulkan loader is missing instance enumeration or creation entry points";
        return false;
    }

    uint32_t extension_count = 0;
    VkResult result = enumerate_extensions(nullptr, &extension_count, nullptr);
    if (result != VK_SUCCESS) {
        error = "vkEnumerateInstanceExtensionProperties failed: " + ResultText(result);
        return false;
    }
    std::vector<VkExtensionProperties> extensions(extension_count);
    result = enumerate_extensions(nullptr, &extension_count, extensions.data());
    if (result != VK_SUCCESS) {
        error = "vkEnumerateInstanceExtensionProperties list failed: " + ResultText(result);
        return false;
    }

    if (!HasExtension(extensions, VK_KHR_SURFACE_EXTENSION_NAME)) {
        error = "Vulkan loader lacks VK_KHR_surface";
        return false;
    }

    std::vector<const char*> enabled_extensions = {
        VK_KHR_SURFACE_EXTENSION_NAME,
    };
    if (enable_win32_wsi_) {
        if (!HasExtension(extensions, kWin32SurfaceExtensionName)) {
            error = "Vulkan loader lacks VK_KHR_win32_surface";
            return false;
        }
        enabled_extensions.push_back(kWin32SurfaceExtensionName);
        if (HasExtension(extensions, VK_EXT_SURFACE_MAINTENANCE_1_EXTENSION_NAME)) {
            enabled_extensions.push_back(VK_EXT_SURFACE_MAINTENANCE_1_EXTENSION_NAME);
            surface_maintenance1_enabled_ = true;
        }
        if (HasExtension(extensions, VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME)) {
            enabled_extensions.push_back(VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME);
            get_surface_capabilities2_enabled_ = true;
        }
    }
    const bool debug_utils = HasExtension(extensions, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    if (debug_utils) enabled_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    std::vector<const char*> enabled_layers;
    if (enumerate_layers) {
        uint32_t layer_count = 0;
        if (enumerate_layers(&layer_count, nullptr) == VK_SUCCESS) {
            std::vector<VkLayerProperties> layers(layer_count);
            if (enumerate_layers(&layer_count, layers.data()) == VK_SUCCESS)
                validation_available_ = HasLayer(layers, "VK_LAYER_KHRONOS_validation");
        }
    }
    if (validation_available_) {
        enabled_layers.push_back("VK_LAYER_KHRONOS_validation");
        validation_enabled_ = true;
    } else {
        LOGW("vulkan: VK_LAYER_KHRONOS_validation is unavailable");
    }

    VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    application.pApplicationName = "CoDF";
    application.applicationVersion = 1;
    application.pEngineName = "CoDF";
    application.engineVersion = 1;
    application.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo create{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    create.pApplicationInfo = &application;
    create.enabledLayerCount = static_cast<uint32_t>(enabled_layers.size());
    create.ppEnabledLayerNames = enabled_layers.data();
    create.enabledExtensionCount = static_cast<uint32_t>(enabled_extensions.size());
    create.ppEnabledExtensionNames = enabled_extensions.data();
    result = create_instance(&create, nullptr, &instance_);
    if (result != VK_SUCCESS) {
        error = "vkCreateInstance failed: " + ResultText(result);
        return false;
    }

    if (auto instance_destroy = InstanceFunction<PFN_vkDestroyInstance>(loader_, instance_, "vkDestroyInstance"))
        destroy_instance_ = instance_destroy;
    enumerate_physical_devices_ =
        InstanceFunction<PFN_vkEnumeratePhysicalDevices>(loader_, instance_, "vkEnumeratePhysicalDevices");
    get_physical_device_properties_ =
        InstanceFunction<PFN_vkGetPhysicalDeviceProperties>(loader_, instance_, "vkGetPhysicalDeviceProperties");
    get_physical_device_properties2_ =
        InstanceFunction<PFN_vkGetPhysicalDeviceProperties2>(loader_, instance_, "vkGetPhysicalDeviceProperties2");
    get_physical_device_features2_ =
        InstanceFunction<PFN_vkGetPhysicalDeviceFeatures2>(loader_, instance_, "vkGetPhysicalDeviceFeatures2");
    get_queue_family_properties_ = InstanceFunction<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
        loader_, instance_, "vkGetPhysicalDeviceQueueFamilyProperties");
    enumerate_device_extensions_ = InstanceFunction<PFN_vkEnumerateDeviceExtensionProperties>(
        loader_, instance_, "vkEnumerateDeviceExtensionProperties");
    get_format_properties_ =
        InstanceFunction<PFN_vkGetPhysicalDeviceFormatProperties>(loader_, instance_, "vkGetPhysicalDeviceFormatProperties");
    get_image_format_properties_ = InstanceFunction<PFN_vkGetPhysicalDeviceImageFormatProperties>(
        loader_, instance_, "vkGetPhysicalDeviceImageFormatProperties");
    create_device_ = InstanceFunction<PFN_vkCreateDevice>(loader_, instance_, "vkCreateDevice");
    if (!destroy_instance_ || !enumerate_physical_devices_ || !get_physical_device_properties_ ||
        !get_physical_device_properties2_ ||
        !get_physical_device_features2_ || !get_queue_family_properties_ ||
        !enumerate_device_extensions_ || !get_format_properties_ || !get_image_format_properties_ || !create_device_) {
        error = "Vulkan instance is missing required device discovery entry points";
        return false;
    }

    if (debug_utils) {
        create_debug_messenger_ = InstanceFunction<PFN_vkCreateDebugUtilsMessengerEXT>(
            loader_, instance_, "vkCreateDebugUtilsMessengerEXT");
        destroy_debug_messenger_ = InstanceFunction<PFN_vkDestroyDebugUtilsMessengerEXT>(
            loader_, instance_, "vkDestroyDebugUtilsMessengerEXT");
        if (create_debug_messenger_ && destroy_debug_messenger_) {
            VkDebugUtilsMessengerCreateInfoEXT debug_create{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
            debug_create.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            debug_create.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                       VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            debug_create.pfnUserCallback = DebugCallback;
            debug_create.pUserData = this;
            const VkResult debug_result = create_debug_messenger_(instance_, &debug_create, nullptr, &debug_messenger_);
            if (debug_result != VK_SUCCESS) {
                LOGW("vulkan: debug messenger creation failed: %s", ResultText(debug_result).c_str());
                debug_messenger_ = VK_NULL_HANDLE;
            }
        }
    } else {
        LOGW("vulkan: VK_EXT_debug_utils is unavailable");
    }
    return true;
}

bool Context::HasDeviceExtension(VkPhysicalDevice physical, const char* name) const {
    uint32_t count = 0;
    if (enumerate_device_extensions_(physical, nullptr, &count, nullptr) != VK_SUCCESS) return false;
    std::vector<VkExtensionProperties> extensions(count);
    if (enumerate_device_extensions_(physical, nullptr, &count, extensions.data()) != VK_SUCCESS) return false;
    return HasExtension(extensions, name);
}

bool Context::CheckFormats(VkPhysicalDevice physical, std::string& error) const {
    const ImageRequirement requirements[] = {
        {VK_FORMAT_R8G8B8A8_UNORM, "R8G8B8A8_UNORM", VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                                                            VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT,
         VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
             VK_IMAGE_USAGE_TRANSFER_DST_BIT},
        {VK_FORMAT_B8G8R8A8_UNORM, "B8G8R8A8_UNORM", VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                                                           VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT,
         VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
             VK_IMAGE_USAGE_TRANSFER_DST_BIT},
        {VK_FORMAT_R16G16B16A16_SFLOAT, "R16G16B16A16_SFLOAT", VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                                                                       VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT,
         VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
             VK_IMAGE_USAGE_TRANSFER_DST_BIT},
        {VK_FORMAT_D24_UNORM_S8_UINT, "D24_UNORM_S8_UINT", VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                                                                  VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT,
         VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
             VK_IMAGE_USAGE_TRANSFER_DST_BIT},
        {VK_FORMAT_D32_SFLOAT_S8_UINT, "D32_SFLOAT_S8_UINT", VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                                                                   VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT,
         VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
             VK_IMAGE_USAGE_TRANSFER_DST_BIT},
    };
    for (const ImageRequirement& requirement : requirements) {
        VkFormatProperties format{};
        get_format_properties_(physical, requirement.format, &format);
        if ((format.optimalTilingFeatures & requirement.features) != requirement.features) {
            error = std::string("Vulkan format lacks required features: ") + requirement.name;
            return false;
        }
        VkImageFormatProperties image{};
        const VkResult result = get_image_format_properties_(physical, requirement.format, VK_IMAGE_TYPE_2D,
                                                             VK_IMAGE_TILING_OPTIMAL, requirement.usage, 0, &image);
        if (result != VK_SUCCESS) {
            error = std::string("Vulkan image usage is unsupported for format ") + requirement.name + ": " +
                    ResultText(result);
            return false;
        }
        if ((image.sampleCounts & VK_SAMPLE_COUNT_4_BIT) == 0) {
            error = std::string("Vulkan format lacks 4x multisampling: ") + requirement.name;
            return false;
        }
    }
    return true;
}

bool Context::CheckPhysicalDevice(VkPhysicalDevice physical, VkPhysicalDeviceProperties& properties,
                                  uint32_t& queue_family, std::string& error) const {
    get_physical_device_properties_(physical, &properties);
    if (!AtLeast13(properties.apiVersion)) {
        error = "Vulkan physical device exposes less than API 1.3";
        return false;
    }
    if (!HasDeviceExtension(physical, VK_KHR_SWAPCHAIN_EXTENSION_NAME)) {
        error = "Vulkan physical device lacks VK_KHR_swapchain";
        return false;
    }

    VkPhysicalDeviceFeatures2 available{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    VkPhysicalDeviceVulkan12Features available12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features available13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    available.pNext = &available12;
    available12.pNext = &available13;
    get_physical_device_features2_(physical, &available);
    const VkPhysicalDeviceFeatures& base = available.features;
    if (!base.geometryShader || !base.tessellationShader || !base.textureCompressionBC || !base.samplerAnisotropy) {
        error = "Vulkan physical device lacks geometry, tessellation, BC compression or anisotropy";
        return false;
    }
    if (!available12.timelineSemaphore || !available13.dynamicRendering || !available13.synchronization2) {
        error = "Vulkan physical device lacks timeline semaphore, dynamic rendering or synchronization2";
        return false;
    }
    if (properties.limits.maxPerStageDescriptorUniformBuffers < kRequiredConstantBuffers ||
        properties.limits.maxDescriptorSetUniformBuffers < kRequiredConstantBuffers ||
        properties.limits.maxDescriptorSetUniformBuffersDynamic < kRequiredConstantBuffers ||
        properties.limits.maxPerStageDescriptorSampledImages < kRequiredSampledImages ||
        properties.limits.maxPerStageDescriptorSamplers < kRequiredSamplers ||
        properties.limits.maxPerStageResources < kRequiredDescriptorsPerStage ||
        properties.limits.maxUniformBufferRange < kRequiredUniformBufferRange) {
        error = "Vulkan descriptor limits are below the shader ABI requirements";
        return false;
    }
    if (!CheckFormats(physical, error)) return false;

    uint32_t queue_count = 0;
    get_queue_family_properties_(physical, &queue_count, nullptr);
    std::vector<VkQueueFamilyProperties> queues(queue_count);
    get_queue_family_properties_(physical, &queue_count, queues.data());
    queue_family = VK_QUEUE_FAMILY_IGNORED;
    for (uint32_t i = 0; i < queue_count; i++) {
        if (queues[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            queue_family = i;
            break;
        }
    }
    if (queue_family == VK_QUEUE_FAMILY_IGNORED) {
        error = "Vulkan physical device has no graphics queue";
        return false;
    }
    return true;
}

bool Context::SelectPhysicalDevice(std::string& error) {
    uint32_t count = 0;
    VkResult result = enumerate_physical_devices_(instance_, &count, nullptr);
    if (result != VK_SUCCESS || count == 0) {
        error = "vkEnumeratePhysicalDevices found no usable devices: " + ResultText(result);
        return false;
    }
    std::vector<VkPhysicalDevice> devices(count);
    result = enumerate_physical_devices_(instance_, &count, devices.data());
    if (result != VK_SUCCESS) {
        error = "vkEnumeratePhysicalDevices list failed: " + ResultText(result);
        return false;
    }

    int best_score = -1;
    std::string last_error = "no device met the Vulkan requirements";
    for (VkPhysicalDevice physical : devices) {
        VkPhysicalDeviceProperties properties{};
        uint32_t queue_family = VK_QUEUE_FAMILY_IGNORED;
        std::string candidate_error;
        if (!CheckPhysicalDevice(physical, properties, queue_family, candidate_error)) {
            last_error = std::move(candidate_error);
            continue;
        }
        const int score = properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 2 : 1;
        if (score > best_score) {
            best_score = score;
            physical_device_ = physical;
            properties_ = properties;
            graphics_queue_family_ = queue_family;
        }
    }
    if (physical_device_ == VK_NULL_HANDLE) {
        error = "No Vulkan physical device satisfies the full PC requirements: " + last_error;
        return false;
    }
    VkPhysicalDeviceFeatures2 selected_features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    VkPhysicalDeviceVulkan12Features selected_features12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features selected_features13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceCustomBorderColorFeaturesEXT selected_custom_border{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT};
    selected_features.pNext = &selected_features12;
    selected_features12.pNext = &selected_features13;
    custom_border_color_extension_available_ =
        HasDeviceExtension(physical_device_, VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME);
    if (custom_border_color_extension_available_) selected_features13.pNext = &selected_custom_border;
    get_physical_device_features2_(physical_device_, &selected_features);
    sampler_mirror_clamp_to_edge_enabled_ = selected_features12.samplerMirrorClampToEdge == VK_TRUE;
    VkPhysicalDeviceProperties2 selected_properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    VkPhysicalDeviceIDProperties selected_id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    selected_properties.pNext = &selected_id;
    VkPhysicalDeviceCustomBorderColorPropertiesEXT custom_properties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_PROPERTIES_EXT};
    if (custom_border_color_extension_available_) selected_id.pNext = &custom_properties;
    get_physical_device_properties2_(physical_device_, &selected_properties);
    std::copy(selected_id.deviceUUID, selected_id.deviceUUID + VK_UUID_SIZE, device_uuid_.begin());
    if (custom_border_color_extension_available_) {
        custom_border_color_features_available_ = selected_custom_border.customBorderColors == VK_TRUE;
        custom_border_color_without_format_available_ =
            selected_custom_border.customBorderColorWithoutFormat == VK_TRUE;
        max_custom_border_color_samplers_ = custom_properties.maxCustomBorderColorSamplers;
    }
    return true;
}

bool Context::CreateDevice(std::string& error) {
    destroy_device_ = InstanceFunction<PFN_vkDestroyDevice>(loader_, instance_, "vkDestroyDevice");
    if (!destroy_device_) {
        error = "Vulkan instance is missing vkDestroyDevice; refusing to create an unteardownable device";
        return false;
    }
    get_device_proc_addr_ = InstanceFunction<PFN_vkGetDeviceProcAddr>(loader_, instance_, "vkGetDeviceProcAddr");
    if (!get_device_proc_addr_) {
        error = "Vulkan instance is missing vkGetDeviceProcAddr; refusing to create a device";
        return false;
    }
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue.queueFamilyIndex = graphics_queue_family_;
    queue.queueCount = 1;
    queue.pQueuePriorities = &priority;

    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features.features.geometryShader = VK_TRUE;
    features.features.tessellationShader = VK_TRUE;
    features.features.textureCompressionBC = VK_TRUE;
    features.features.samplerAnisotropy = VK_TRUE;
    VkPhysicalDeviceVulkan12Features features12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    features12.timelineSemaphore = VK_TRUE;
    features12.samplerMirrorClampToEdge = sampler_mirror_clamp_to_edge_enabled_ ? VK_TRUE : VK_FALSE;
    VkPhysicalDeviceVulkan13Features features13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    features13.dynamicRendering = VK_TRUE;
    features13.synchronization2 = VK_TRUE;
    VkPhysicalDeviceFeatures2 available{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    VkPhysicalDeviceVulkan12Features available12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features available13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceCustomBorderColorFeaturesEXT available_custom_border{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT};
    VkPhysicalDeviceSwapchainMaintenance1FeaturesEXT available_maintenance{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_EXT};
    VkPhysicalDeviceCustomBorderColorFeaturesEXT enabled_custom_border{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT};
    VkPhysicalDeviceSwapchainMaintenance1FeaturesEXT enabled_maintenance{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_EXT};
    bool enable_swapchain_maintenance1 = false;
    const bool enable_custom_border_color = custom_border_color_extension_available_ &&
                                            custom_border_color_features_available_;
    if (enable_custom_border_color ||
        (enable_win32_wsi_ && surface_maintenance1_enabled_ &&
         HasDeviceExtension(physical_device_, VK_EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME))) {
        available.pNext = &available12;
        available12.pNext = &available13;
        available13.pNext = nullptr;
    }
    if (enable_win32_wsi_ && surface_maintenance1_enabled_ &&
        HasDeviceExtension(physical_device_, VK_EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME)) {
        available13.pNext = &available_maintenance;
        if (enable_custom_border_color) available_maintenance.pNext = &available_custom_border;
        get_physical_device_features2_(physical_device_, &available);
        enable_swapchain_maintenance1 = available_maintenance.swapchainMaintenance1 == VK_TRUE;
    } else if (enable_custom_border_color) {
        available13.pNext = &available_custom_border;
        get_physical_device_features2_(physical_device_, &available);
    }
    features.pNext = &features12;
    features12.pNext = &features13;
    enabled_custom_border.customBorderColors = enable_custom_border_color ? VK_TRUE : VK_FALSE;
    enabled_custom_border.customBorderColorWithoutFormat =
        enable_custom_border_color && custom_border_color_without_format_available_ ? VK_TRUE : VK_FALSE;
    if (enable_swapchain_maintenance1) {
        enabled_maintenance.swapchainMaintenance1 = VK_TRUE;
        if (enable_custom_border_color) {
            enabled_maintenance.pNext = &enabled_custom_border;
            features13.pNext = &enabled_maintenance;
        } else {
            features13.pNext = &enabled_maintenance;
        }
    } else if (enable_custom_border_color) {
        features13.pNext = &enabled_custom_border;
    }

    std::vector<const char*> extensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    if (enable_swapchain_maintenance1) extensions.push_back(VK_EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME);
    if (enable_custom_border_color) extensions.push_back(VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME);
    VkDeviceCreateInfo create{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    create.pNext = &features;
    create.queueCreateInfoCount = 1;
    create.pQueueCreateInfos = &queue;
    create.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    create.ppEnabledExtensionNames = extensions.data();
    const VkResult result = create_device_(physical_device_, &create, nullptr, &device_);
    if (result != VK_SUCCESS) {
        error = "vkCreateDevice failed: " + ResultText(result);
        return false;
    }
    swapchain_maintenance1_enabled_ = enable_swapchain_maintenance1;
    custom_border_color_enabled_ = enable_custom_border_color;
    custom_border_color_without_format_enabled_ =
        enable_custom_border_color && custom_border_color_without_format_available_;

    if (auto device_destroy = reinterpret_cast<PFN_vkDestroyDevice>(get_device_proc_addr_(device_, "vkDestroyDevice")))
        destroy_device_ = device_destroy;
    device_wait_idle_ = reinterpret_cast<PFN_vkDeviceWaitIdle>(get_device_proc_addr_(device_, "vkDeviceWaitIdle"));
    get_device_queue_ = reinterpret_cast<PFN_vkGetDeviceQueue>(get_device_proc_addr_(device_, "vkGetDeviceQueue"));
    if (!destroy_device_ || !device_wait_idle_ || !get_device_queue_) {
        error = "Vulkan device is missing required queue or teardown entry points";
        return false;
    }
    get_device_queue_(device_, graphics_queue_family_, 0, &graphics_queue_);
    if (graphics_queue_ == VK_NULL_HANDLE) {
        error = "Vulkan returned no graphics queue";
        return false;
    }
    pipeline_cache_.Init(*this, pipeline_cache_directory_);
    return true;
}

VKAPI_ATTR VkBool32 VKAPI_CALL Context::DebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                       VkDebugUtilsMessageTypeFlagsEXT type,
                                                       const VkDebugUtilsMessengerCallbackDataEXT* data,
                                                       void* user_data) {
    (void)type;
    auto* context = static_cast<Context*>(user_data);
    const char* message = data && data->pMessage ? data->pMessage : "(no Vulkan debug message)";
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        if (context) context->debug_error_count_.fetch_add(1, std::memory_order_relaxed);
        if (context && context->debug_log_count_.fetch_add(1, std::memory_order_relaxed) < kMaxDebugLogMessages)
            LOGE("vulkan: %s", message);
    } else {
        if (context) context->debug_warning_count_.fetch_add(1, std::memory_order_relaxed);
        if (context && context->debug_log_count_.fetch_add(1, std::memory_order_relaxed) < kMaxDebugLogMessages)
            LOGW("vulkan: %s", message);
    }
    return VK_FALSE;
}

}  // namespace gpu::vulkan
