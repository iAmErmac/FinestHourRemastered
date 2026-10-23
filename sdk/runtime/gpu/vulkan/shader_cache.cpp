#include "gpu/vulkan/shader_cache.h"

#include "gpu/shaders.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <thread>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace gpu::vulkan {
namespace {

constexpr uint32_t kMagic = 0x31565343;  // CSV1
constexpr uint32_t kFormatVersion = 1;
constexpr uint32_t kSpirvMagic = 0x07230203;
constexpr size_t kMaxSpirvBytes = 16u * 1024u * 1024u;
constexpr size_t kMaxCacheEntries = 4096;
constexpr uintmax_t kMaxCacheBytes = 256u * 1024u * 1024u;
constexpr uint64_t kAlternateHashSeed = 0xD6E8FEB86659FD93ull;

#pragma pack(push, 1)
struct CacheHeader {
    uint32_t magic;
    uint32_t format_version;
    uint32_t header_size;
    uint32_t stage;
    uint32_t shader_abi;
    uint32_t source_schema;
    uint32_t reserved;
    uint64_t source_size;
    uint64_t source_hash;
    uint64_t source_hash_alt;
    uint64_t entry_size;
    uint64_t entry_hash;
    uint64_t compiler_identity;
    uint64_t options_identity;
    uint64_t binding_identity;
    uint64_t key_digest;
    uint64_t spirv_words;
    uint64_t spirv_hash;
};
#pragma pack(pop)

static_assert(sizeof(CacheHeader) == 116);

uint64_t HashString(const std::string& value, uint64_t seed = 1469598103934665603ull) {
    return gpu::Hash(value.data(), value.size(), seed);
}

uint64_t HashKey(const ShaderCacheKey& key) {
    uint64_t result = 1469598103934665603ull;
    auto add = [&](uint64_t value) { result = gpu::Hash(&value, sizeof(value), result); };
    add(key.stage);
    add(key.shader_abi);
    add(key.source_schema);
    add(key.compiler_identity);
    add(key.options_identity);
    add(key.binding_identity);
    add(static_cast<uint64_t>(key.source.size()));
    add(HashString(key.source));
    add(HashString(key.source, kAlternateHashSeed));
    add(static_cast<uint64_t>(key.entry.size()));
    add(HashString(key.entry));
    return result;
}

CacheHeader MakeHeader(const ShaderCacheKey& key, const std::vector<uint32_t>& spirv) {
    CacheHeader header{};
    header.magic = kMagic;
    header.format_version = kFormatVersion;
    header.header_size = sizeof(CacheHeader);
    header.stage = key.stage;
    header.shader_abi = key.shader_abi;
    header.source_schema = key.source_schema;
    header.source_size = key.source.size();
    header.source_hash = HashString(key.source);
    header.source_hash_alt = HashString(key.source, kAlternateHashSeed);
    header.entry_size = key.entry.size();
    header.entry_hash = HashString(key.entry);
    header.compiler_identity = key.compiler_identity;
    header.options_identity = key.options_identity;
    header.binding_identity = key.binding_identity;
    header.key_digest = HashKey(key);
    header.spirv_words = spirv.size();
    header.spirv_hash = gpu::Hash(spirv.data(), spirv.size() * sizeof(uint32_t));
    return header;
}

bool MatchesKey(const CacheHeader& header, const ShaderCacheKey& key) {
    return header.magic == kMagic && header.format_version == kFormatVersion &&
           header.header_size == sizeof(CacheHeader) && header.stage == key.stage &&
           header.shader_abi == key.shader_abi && header.source_schema == key.source_schema &&
           header.source_size == key.source.size() && header.source_hash == HashString(key.source) &&
           header.source_hash_alt == HashString(key.source, kAlternateHashSeed) &&
           header.entry_size == key.entry.size() && header.entry_hash == HashString(key.entry) &&
           header.compiler_identity == key.compiler_identity && header.options_identity == key.options_identity &&
           header.binding_identity == key.binding_identity && header.key_digest == HashKey(key);
}

std::filesystem::path CachePath(const std::filesystem::path& directory, const ShaderCacheKey& key) {
    char name[48];
    std::snprintf(name, sizeof(name), "%016llx.spvc", static_cast<unsigned long long>(HashKey(key)));
    return directory / name;
}

bool CacheBudgetAvailable(const std::filesystem::path& directory, const std::filesystem::path& destination,
                          uintmax_t incoming_bytes) {
    if (incoming_bytes > kMaxCacheBytes) return false;
    std::error_code error;
    size_t entries = 0;
    uintmax_t bytes = 0;
    bool replacing = false;
    for (std::filesystem::directory_iterator it(directory, error), end; it != end && !error; it.increment(error)) {
        const std::filesystem::directory_entry& entry = *it;
        if (entry.path().extension() != ".spvc") continue;
        if (!entry.is_regular_file(error) || error) return false;
        const uintmax_t size = entry.file_size(error);
        if (error) return false;
        if (entry.path() == destination) {
            replacing = true;
            continue;
        }
        ++entries;
        if (kMaxCacheBytes - bytes < size) return false;
        bytes += size;
    }
    if (error) return false;
    if (!replacing && entries >= kMaxCacheEntries) return false;
    return bytes <= kMaxCacheBytes - incoming_bytes;
}

std::filesystem::path TempPath(const std::filesystem::path& path) {
    std::ostringstream suffix;
    suffix << ".tmp-" << std::hex << std::hash<std::thread::id>{}(std::this_thread::get_id());
#if defined(_WIN32)
    suffix << '-' << std::hex << GetCurrentProcessId();
#endif
    std::filesystem::path temporary = path;
    temporary += suffix.str();
    return temporary;
}

bool ReplaceAtomically(const std::filesystem::path& temporary, const std::filesystem::path& destination) {
#if defined(_WIN32)
    return MoveFileExW(temporary.c_str(), destination.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    std::error_code error;
    std::filesystem::rename(temporary, destination, error);
    return !error;
#endif
}

}  // namespace

void ShaderCache::SetDirectory(std::filesystem::path directory) {
    std::lock_guard lock(mutex_);
    directory_ = std::move(directory);
}

bool ShaderCache::Enabled() const {
    std::lock_guard lock(mutex_);
    return !directory_.empty();
}

bool ShaderCache::Load(const ShaderCacheKey& key, std::vector<uint32_t>& spirv) const {
    std::lock_guard lock(mutex_);
    spirv.clear();
    if (directory_.empty()) return false;
    const std::filesystem::path path = CachePath(directory_, key);
    std::error_code error;
    const uintmax_t file_size = std::filesystem::file_size(path, error);
    if (error || file_size < sizeof(CacheHeader) || file_size > sizeof(CacheHeader) + kMaxSpirvBytes) return false;

    std::ifstream input(path, std::ios::binary);
    if (!input) return false;
    CacheHeader header{};
    input.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!input || !MatchesKey(header, key) || !header.spirv_words || header.spirv_words > kMaxSpirvBytes / 4 ||
        header.spirv_words * sizeof(uint32_t) != file_size - sizeof(CacheHeader))
        return false;

    spirv.resize(static_cast<size_t>(header.spirv_words));
    input.read(reinterpret_cast<char*>(spirv.data()), static_cast<std::streamsize>(spirv.size() * sizeof(uint32_t)));
    if (!input || spirv.empty() || spirv[0] != kSpirvMagic ||
        gpu::Hash(spirv.data(), spirv.size() * sizeof(uint32_t)) != header.spirv_hash) {
        spirv.clear();
        return false;
    }
    return true;
}

bool ShaderCache::Store(const ShaderCacheKey& key, const std::vector<uint32_t>& spirv) {
    if (spirv.empty() || spirv[0] != kSpirvMagic || spirv.size() > kMaxSpirvBytes / sizeof(uint32_t)) return false;
    std::lock_guard lock(mutex_);
    if (directory_.empty()) return false;
    std::error_code error;
    std::filesystem::create_directories(directory_, error);
    if (error) return false;

    const std::filesystem::path path = CachePath(directory_, key);
    const uintmax_t incoming_bytes = sizeof(CacheHeader) + spirv.size() * sizeof(uint32_t);
    if (!CacheBudgetAvailable(directory_, path, incoming_bytes)) return false;
    const std::filesystem::path temporary = TempPath(path);
    const CacheHeader header = MakeHeader(key, spirv);
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) return false;
        output.write(reinterpret_cast<const char*>(&header), sizeof(header));
        output.write(reinterpret_cast<const char*>(spirv.data()),
                     static_cast<std::streamsize>(spirv.size() * sizeof(uint32_t)));
        output.flush();
        if (!output) {
            output.close();
            std::filesystem::remove(temporary, error);
            return false;
        }
    }
    if (!ReplaceAtomically(temporary, path)) {
        std::filesystem::remove(temporary, error);
        return false;
    }
    ++stats_.stores;
    return true;
}

void ShaderCache::RecordHit() const {
    std::lock_guard lock(mutex_);
    ++stats_.hits;
}

void ShaderCache::RecordMiss() const {
    std::lock_guard lock(mutex_);
    ++stats_.misses;
}

void ShaderCache::RecordCompilerInvocation() const {
    std::lock_guard lock(mutex_);
    ++stats_.compiler_invocations;
}

void ShaderCache::RecordRejected() const {
    std::lock_guard lock(mutex_);
    ++stats_.rejected;
}

ShaderCacheStats ShaderCache::Stats() const {
    std::lock_guard lock(mutex_);
    return stats_;
}

}  // namespace gpu::vulkan
