#pragma once

#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace gpu::vulkan {

// The source is retained in the key only while a cache operation is in progress. It is
// hashed into the on-disk identity; the full HLSL is never written to the cache.
struct ShaderCacheKey {
    uint32_t stage = 0;
    uint32_t shader_abi = 0;
    uint32_t source_schema = 0;
    uint32_t reserved = 0;
    uint64_t compiler_identity = 0;
    uint64_t options_identity = 0;
    uint64_t binding_identity = 0;
    std::string source;
    std::string entry;
};

struct ShaderCacheStats {
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t compiler_invocations = 0;
    uint64_t stores = 0;
    uint64_t rejected = 0;
};

// Persistent compiled-SPIR-V storage. It has no app/frontend dependency so compiler
// fixtures can use it with an explicit directory, while the runtime supplies its own
// private per-backend directory.
class ShaderCache {
public:
    ShaderCache() = default;
    explicit ShaderCache(std::filesystem::path directory) : directory_(std::move(directory)) {}
    ShaderCache(const ShaderCache&) = delete;
    ShaderCache& operator=(const ShaderCache&) = delete;

    void SetDirectory(std::filesystem::path directory);
    bool Enabled() const;

    // Returns true only for a structurally valid record whose key matches exactly. The
    // caller must still reflect/validate the module before accepting it as a hit.
    bool Load(const ShaderCacheKey& key, std::vector<uint32_t>& spirv) const;
    bool Store(const ShaderCacheKey& key, const std::vector<uint32_t>& spirv);

    void RecordHit() const;
    void RecordMiss() const;
    void RecordCompilerInvocation() const;
    void RecordRejected() const;
    ShaderCacheStats Stats() const;

private:
    std::filesystem::path directory_;
    mutable std::mutex mutex_;
    mutable ShaderCacheStats stats_;
};

}  // namespace gpu::vulkan
