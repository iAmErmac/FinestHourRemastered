#pragma once

#include "gpu/shaders.h"

#include <cstdint>
#include <string>
#include <unordered_map>

namespace gpu::vulkan {

struct ShaderSourceRecipe {
    uint64_t program_hash = 0;
    uint32_t program_words = 0;
    uint32_t declaration_handle = 0;
    uint16_t regs = 0;
    uint64_t decode = 0;
    bool programmable = false;
    bool pretransformed = false;
    bool point_geometry = false;
    FixedKey fixed{};
    PixelKey pixel{};
    const ShaderHooks* hooks = nullptr;

    bool operator==(const ShaderSourceRecipe& other) const {
        return program_hash == other.program_hash && program_words == other.program_words &&
               declaration_handle == other.declaration_handle && regs == other.regs && decode == other.decode &&
               programmable == other.programmable && pretransformed == other.pretransformed &&
               point_geometry == other.point_geometry && fixed == other.fixed && pixel == other.pixel &&
               hooks == other.hooks;
    }
};

struct ShaderSourceRecipeHash {
    size_t operator()(const ShaderSourceRecipe& recipe) const;
};

struct DrawSources {
    std::string vertex;
    std::string pixel;
    std::string geometry;
};

class DrawSourceCache {
public:
    const DrawSources& Get(VertexShader* vertex, uint32_t vs_handle, uint16_t regs, uint64_t decode,
                            bool pretransformed, const FixedKey& fixed, const PixelKey& pixel, bool point_geometry,
                            bool* cache_hit = nullptr);

    void Clear();

private:
    std::unordered_map<ShaderSourceRecipe, DrawSources, ShaderSourceRecipeHash> sources_;
};

}  // namespace gpu::vulkan
