#include "gpu/vulkan/pipeline_cache.h"

#include "core/log.h"
#include "gpu/shaders.h"
#include "gpu/vulkan/context.h"

#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace gpu::vulkan {
namespace {

constexpr uint32_t kMagic = 0x31504343;  // CCP1
constexpr uint32_t kFormatVersion = 1;
constexpr uint64_t kMaxCacheBytes = 64ull * 1024ull * 1024ull;
constexpr uint64_t kAlternateHashSeed = 1099511628211ull ^ 0x9e3779b97f4a7c15ull;

#pragma pack(push, 1)
struct CacheHeader {
    uint32_t magic;
    uint32_t format_version;
    uint32_t header_size;
    uint32_t payload_size;
    uint32_t vendor_id;
    uint32_t device_id;
    uint32_t driver_version;
    uint8_t pipeline_cache_uuid[VK_UUID_SIZE];
    uint8_t device_uuid[VK_UUID_SIZE];
    uint64_t payload_hash;
    uint64_t payload_hash_alt;
};
#pragma pack(pop)

static_assert(sizeof(CacheHeader) == 76);

uint64_t HashBytes(const void* data, size_t size, uint64_t seed = 1469598103934665603ull) {
    return gpu::Hash(data, size, seed);
}

std::filesystem::path CachePath(const std::filesystem::path& directory) {
    return directory / "pipeline_cache.vkpc";
}

std::filesystem::path TempPath(const std::filesystem::path& path) {
    const uint64_t thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
#ifdef _WIN32
    const uint64_t process_id = GetCurrentProcessId();
#else
    const uint64_t process_id = 0;
#endif
    std::filesystem::path temporary = path;
    temporary += ".tmp-" + std::to_string(process_id) + "-" + std::to_string(thread_id);
    return temporary;
}

bool ReadFile(const std::filesystem::path& path, std::vector<uint8_t>& bytes) {
    bytes.clear();
    std::error_code error;
    const uintmax_t size = std::filesystem::file_size(path, error);
    if (error || size < sizeof(CacheHeader) || size > sizeof(CacheHeader) + kMaxCacheBytes) return false;
    bytes.resize(static_cast<size_t>(size));
    std::ifstream file(path, std::ios::binary);
    if (!file || !file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()))) {
        bytes.clear();
        return false;
    }
    return true;
}

std::string ResultText(VkResult result) {
    return "VkResult " + std::to_string(static_cast<int>(result));
}

bool AtomicReplace(const std::filesystem::path& temporary, const std::filesystem::path& destination) {
#ifdef _WIN32
    return MoveFileExW(temporary.c_str(), destination.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    std::error_code error;
    std::filesystem::rename(temporary, destination, error);
    return !error;
#endif
}

CacheHeader MakeHeader(const Context& context, const std::vector<uint8_t>& payload) {
    CacheHeader header{};
    header.magic = kMagic;
    header.format_version = kFormatVersion;
    header.header_size = sizeof(CacheHeader);
    header.payload_size = static_cast<uint32_t>(payload.size());
    header.vendor_id = context.Properties().vendorID;
    header.device_id = context.Properties().deviceID;
    header.driver_version = context.Properties().driverVersion;
    std::memcpy(header.pipeline_cache_uuid, context.Properties().pipelineCacheUUID, VK_UUID_SIZE);
    std::memcpy(header.device_uuid, context.DeviceUUID().data(), VK_UUID_SIZE);
    header.payload_hash = HashBytes(payload.data(), payload.size());
    header.payload_hash_alt = HashBytes(payload.data(), payload.size(), kAlternateHashSeed);
    return header;
}

bool Compatible(const Context& context, const CacheHeader& header, const std::vector<uint8_t>& payload) {
    if (header.magic != kMagic || header.format_version != kFormatVersion ||
        header.header_size != sizeof(CacheHeader) || header.payload_size != payload.size() ||
        header.payload_size == 0 || header.payload_size > kMaxCacheBytes ||
        header.vendor_id != context.Properties().vendorID || header.device_id != context.Properties().deviceID ||
        header.driver_version != context.Properties().driverVersion ||
        std::memcmp(header.pipeline_cache_uuid, context.Properties().pipelineCacheUUID, VK_UUID_SIZE) != 0 ||
        std::memcmp(header.device_uuid, context.DeviceUUID().data(), VK_UUID_SIZE) != 0 ||
        header.payload_hash != HashBytes(payload.data(), payload.size()) ||
        header.payload_hash_alt != HashBytes(payload.data(), payload.size(), kAlternateHashSeed))
        return false;
    if (payload.size() < sizeof(VkPipelineCacheHeaderVersionOne)) return false;

    VkPipelineCacheHeaderVersionOne standard{};
    std::memcpy(&standard, payload.data(), sizeof(standard));
    return standard.headerSize >= sizeof(standard) && standard.headerSize <= payload.size() &&
           standard.headerVersion == VK_PIPELINE_CACHE_HEADER_VERSION_ONE &&
           standard.vendorID == context.Properties().vendorID &&
           standard.deviceID == context.Properties().deviceID &&
           std::memcmp(standard.pipelineCacheUUID, context.Properties().pipelineCacheUUID, VK_UUID_SIZE) == 0;
}

bool WriteRecord(const std::filesystem::path& path, const CacheHeader& header, const std::vector<uint8_t>& payload) {
    std::error_code error;
    std::filesystem::create_directories(path.parent_path(), error);
    if (error) return false;
    const std::filesystem::path temporary = TempPath(path);
    {
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        if (!file) return false;
        file.write(reinterpret_cast<const char*>(&header), sizeof(header));
        file.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
        file.flush();
        if (!file) {
            file.close();
            std::filesystem::remove(temporary, error);
            return false;
        }
    }
    if (!AtomicReplace(temporary, path)) {
        std::filesystem::remove(temporary, error);
        return false;
    }
    return true;
}

bool ReadCompatiblePayload(const Context& context, const std::filesystem::path& path,
                           std::vector<uint8_t>& payload) {
    std::vector<uint8_t> bytes;
    if (!ReadFile(path, bytes) || bytes.size() < sizeof(CacheHeader)) return false;
    CacheHeader header{};
    std::memcpy(&header, bytes.data(), sizeof(header));
    if (header.payload_size != bytes.size() - sizeof(CacheHeader)) return false;
    payload.assign(bytes.begin() + sizeof(CacheHeader), bytes.end());
    if (!Compatible(context, header, payload)) {
        payload.clear();
        return false;
    }
    return true;
}

bool SavePayload(const Context& context, const std::filesystem::path& path, PFN_vkGetPipelineCacheData get_data,
                 VkPipelineCache cache, uint64_t& stored_bytes) {
    stored_bytes = 0;
    std::vector<uint8_t> payload;
    for (uint32_t attempt = 0; attempt < 3; ++attempt) {
        size_t size = 0;
        const VkResult size_result = get_data(context.Device(), cache, &size, nullptr);
        if (size_result != VK_SUCCESS && size_result != VK_INCOMPLETE) return false;
        if (!size || size > kMaxCacheBytes) return false;
        payload.resize(size);
        const VkResult data_result = get_data(context.Device(), cache, &size, payload.data());
        if (data_result == VK_SUCCESS) {
            if (!size || size > kMaxCacheBytes) return false;
            payload.resize(size);
            const CacheHeader header = MakeHeader(context, payload);
            if (!WriteRecord(path, header, payload)) return false;
            stored_bytes = size;
            return true;
        }
        if (data_result != VK_INCOMPLETE) return false;
        payload.clear();
    }
    return false;
}

}  // namespace

PipelineCache::~PipelineCache() {
    Reset();
}

void PipelineCache::Init(const Context& context, const std::filesystem::path& directory) {
    Reset();
    stats_ = {};
    if (directory.empty() || context.Device() == VK_NULL_HANDLE) return;
    path_ = CachePath(directory);
    context_ = &context;
    device_ = context.Device();
    auto create = reinterpret_cast<PFN_vkCreatePipelineCache>(context.DeviceProc("vkCreatePipelineCache"));
    destroy_ = reinterpret_cast<PFN_vkDestroyPipelineCache>(context.DeviceProc("vkDestroyPipelineCache"));
    get_data_ = reinterpret_cast<PFN_vkGetPipelineCacheData>(context.DeviceProc("vkGetPipelineCacheData"));
    if (!create || !destroy_ || !get_data_) {
        LOGW("vulkan: pipeline cache entry points unavailable; continuing without persistent pipeline cache");
        Reset();
        return;
    }

    std::vector<uint8_t> payload;
    const bool compatible = ReadCompatiblePayload(context, path_, payload);
    VkPipelineCacheCreateInfo create_info{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    create_info.initialDataSize = compatible ? payload.size() : 0;
    create_info.pInitialData = compatible ? payload.data() : nullptr;
    const bool initial_data_submitted = compatible;
    VkResult result = create(context.Device(), &create_info, nullptr, &cache_);
    bool initial_data_accepted = result == VK_SUCCESS && initial_data_submitted;
    if (result != VK_SUCCESS && compatible) {
        LOGW("vulkan: pipeline cache initial data rejected (%s); retrying with an empty cache", ResultText(result).c_str());
        cache_ = VK_NULL_HANDLE;
        create_info.initialDataSize = 0;
        create_info.pInitialData = nullptr;
        result = create(context.Device(), &create_info, nullptr, &cache_);
        initial_data_accepted = false;
    }
    if (result != VK_SUCCESS) {
        cache_ = VK_NULL_HANDLE;
        LOGW("vulkan: empty pipeline cache creation failed (%s); continuing without persistent pipeline cache",
             ResultText(result).c_str());
        Reset();
        return;
    }
    stats_.enabled = true;
    stats_.loaded_from_disk = initial_data_accepted;
    stats_.loaded_bytes = initial_data_accepted ? payload.size() : 0;
}

bool PipelineCache::Persist() {
    if (cache_ == VK_NULL_HANDLE || !context_ || !get_data_ || path_.empty()) return false;
    uint64_t stored_bytes = 0;
    if (!SavePayload(*context_, path_, get_data_, cache_, stored_bytes)) return false;
    stats_.stored_to_disk = true;
    stats_.stored_bytes = stored_bytes;
    return true;
}

void PipelineCache::Reset(bool persist) {
    if (persist && cache_ != VK_NULL_HANDLE && get_data_ && !path_.empty()) {
        uint64_t stored_bytes = 0;
        if (context_ && SavePayload(*context_, path_, get_data_, cache_, stored_bytes)) {
            stats_.stored_to_disk = true;
            stats_.stored_bytes = stored_bytes;
        }
    }
    if (cache_ != VK_NULL_HANDLE && destroy_) destroy_(device_, cache_, nullptr);
    context_ = nullptr;
    device_ = VK_NULL_HANDLE;
    cache_ = VK_NULL_HANDLE;
    destroy_ = nullptr;
    get_data_ = nullptr;
    path_.clear();
}

}  // namespace gpu::vulkan
