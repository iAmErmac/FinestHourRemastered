#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace gpu::vulkan {

inline constexpr uint32_t kVertexRegisterCount = 192;
inline constexpr uint32_t kVertexRegisterBytes = kVertexRegisterCount * 16u;

// This ABI is shared by the HLSL generator and Vulkan descriptor layouts.
inline constexpr uint32_t kShaderAbiVersion = 1;
inline constexpr uint32_t kConstantBindingBase = 0;
inline constexpr uint32_t kConstantBindingCount = 8;
inline constexpr uint32_t kTextureBindingBase = 16;
inline constexpr uint32_t kTextureBindingCount = 16;
inline constexpr uint32_t kSamplerBindingBase = 32;
inline constexpr uint32_t kSamplerBindingCount = 16;

enum class ShaderStage : uint8_t {
    Vertex,
    Pixel,
    Geometry,
    Hull,
    Domain,
};

struct ShaderBinding {
    enum class Kind : uint8_t {
        Constant,
        Texture,
        Sampler,
    };

    uint32_t set = 0;
    uint32_t binding = 0;
    Kind kind = Kind::Constant;
    std::string name;
};

struct ShaderLocation {
    uint32_t location = 0;
    bool input = false;
    std::string name;
};

struct UniformMemberOffset {
    uint32_t binding = 0;
    uint32_t member = 0;
    uint32_t offset = 0;
    std::string block;
};

struct ShaderReflection {
    ShaderStage stage = ShaderStage::Vertex;
    std::string entry;
    std::vector<ShaderBinding> bindings;
    std::vector<ShaderLocation> locations;
    std::vector<UniformMemberOffset> uniform_member_offsets;
};

struct ShaderModule {
    std::vector<uint32_t> spirv;
    ShaderReflection reflection;
};

bool CompileShader(const std::string& hlsl, const char* entry, ShaderStage stage, ShaderModule& module,
                   std::string& error);

// Parses the generated module and verifies the reflection data used by the host ABI.
bool ReflectShader(const std::vector<uint32_t>& spirv, ShaderReflection& reflection, std::string& error);

}  // namespace gpu::vulkan
