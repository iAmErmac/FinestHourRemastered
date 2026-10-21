#include "gpu/vulkan/shader.h"

#ifndef ENABLE_HLSL
#define ENABLE_HLSL 1
#endif

#include <SPIRV/GlslangToSpv.h>
#include <glslang/Public/ResourceLimits.h>
#include <glslang/Public/ShaderLang.h>

#include <algorithm>
#include <mutex>
#include <regex>
#include <utility>
#include <unordered_set>
#include <unordered_map>

namespace gpu::vulkan {
namespace {

constexpr uint32_t kSpirvMagic = 0x07230203;
constexpr uint16_t kOpName = 5;
constexpr uint16_t kOpEntryPoint = 15;
constexpr uint16_t kOpTypeVoid = 19;
constexpr uint16_t kOpTypeBool = 20;
constexpr uint16_t kOpTypeInt = 21;
constexpr uint16_t kOpTypeFloat = 22;
constexpr uint16_t kOpTypeVector = 23;
constexpr uint16_t kOpTypeMatrix = 24;
constexpr uint16_t kOpTypeImage = 25;
constexpr uint16_t kOpTypeSampler = 26;
constexpr uint16_t kOpTypeSampledImage = 27;
constexpr uint16_t kOpTypeArray = 28;
constexpr uint16_t kOpTypeRuntimeArray = 29;
constexpr uint16_t kOpTypeStruct = 30;
constexpr uint16_t kOpTypePointer = 32;
constexpr uint16_t kOpVariable = 59;
constexpr uint16_t kOpDecorate = 71;
constexpr uint16_t kOpMemberDecorate = 72;
constexpr uint32_t kStorageInput = 1;
constexpr uint32_t kStorageOutput = 3;
constexpr uint32_t kStorageUniform = 2;
constexpr uint32_t kStorageUniformConstant = 0;
constexpr uint32_t kExecutionVertex = 0;
constexpr uint32_t kExecutionTessellationControl = 1;
constexpr uint32_t kExecutionTessellationEvaluation = 2;
constexpr uint32_t kExecutionGeometry = 3;
constexpr uint32_t kExecutionFragment = 4;
constexpr uint32_t kDecorationLocation = 30;
constexpr uint32_t kDecorationBlock = 2;
constexpr uint32_t kDecorationBinding = 33;
constexpr uint32_t kDecorationDescriptorSet = 34;
constexpr uint32_t kDecorationOffset = 35;

struct Decorations {
    bool has_binding = false;
    uint32_t binding = 0;
    bool has_set = false;
    uint32_t set = 0;
    bool has_location = false;
    uint32_t location = 0;
    bool has_block = false;
};

struct TypeInfo {
    uint16_t opcode = 0;
    uint32_t pointee = 0;
    bool has_sampled_operand = false;
    uint32_t sampled_operand = 0;
    std::vector<uint32_t> members;
};

struct Variable {
    uint32_t type = 0;
    uint32_t id = 0;
    uint32_t storage = 0;
};

struct MemberOffset {
    uint32_t type = 0;
    uint32_t member = 0;
    uint32_t offset = 0;
};

enum class DescriptorKind : uint8_t {
    Constant,
    Texture,
    Sampler,
    Unsupported,
};

const char* StageName(ShaderStage stage) {
    switch (stage) {
        case ShaderStage::Vertex: return "vertex";
        case ShaderStage::Pixel: return "pixel";
        case ShaderStage::Geometry: return "geometry";
        case ShaderStage::Hull: return "hull";
        case ShaderStage::Domain: return "domain";
    }
    return "unknown";
}

EShLanguage Language(ShaderStage stage) {
    switch (stage) {
        case ShaderStage::Vertex: return EShLangVertex;
        case ShaderStage::Pixel: return EShLangFragment;
        case ShaderStage::Geometry: return EShLangGeometry;
        case ShaderStage::Hull: return EShLangTessControl;
        case ShaderStage::Domain: return EShLangTessEvaluation;
    }
    return EShLangVertex;
}

bool DecodeStage(uint32_t execution, ShaderStage& stage) {
    switch (execution) {
        case kExecutionVertex: stage = ShaderStage::Vertex; return true;
        case kExecutionTessellationControl: stage = ShaderStage::Hull; return true;
        case kExecutionTessellationEvaluation: stage = ShaderStage::Domain; return true;
        case kExecutionGeometry: stage = ShaderStage::Geometry; return true;
        case kExecutionFragment: stage = ShaderStage::Pixel; return true;
        default: return false;
    }
}

std::string SpvString(const uint32_t* words, uint32_t count, uint32_t start) {
    std::string result;
    for (uint32_t i = start; i < count; ++i) {
        const uint32_t word = words[i];
        for (uint32_t byte = 0; byte < 4; ++byte) {
            const char c = static_cast<char>((word >> (byte * 8)) & 0xff);
            if (!c) return result;
            result.push_back(c);
        }
    }
    return result;
}

DescriptorKind DescriptorType(uint32_t type_id, const std::unordered_map<uint32_t, TypeInfo>& types,
                              const std::unordered_map<uint32_t, Decorations>& decorations) {
    auto type_it = types.find(type_id);
    while (type_it != types.end() &&
           (type_it->second.opcode == kOpTypeArray || type_it->second.opcode == kOpTypeRuntimeArray)) {
        type_it = types.find(type_it->second.pointee);
    }
    if (type_it == types.end()) return DescriptorKind::Unsupported;
    if (type_it->second.opcode == kOpTypeSampler) return DescriptorKind::Sampler;
    if (type_it->second.opcode == kOpTypeImage)
        return type_it->second.has_sampled_operand && type_it->second.sampled_operand == 1
                   ? DescriptorKind::Texture
                   : DescriptorKind::Unsupported;
    if (type_it->second.opcode == kOpTypeSampledImage) return DescriptorKind::Unsupported;
    if (type_it->second.opcode == kOpTypeStruct) {
        const auto decoration = decorations.find(type_it->first);
        return decoration != decorations.end() && decoration->second.has_block ? DescriptorKind::Constant
                                                                                : DescriptorKind::Unsupported;
    }
    return DescriptorKind::Unsupported;
}

bool BindingInRange(DescriptorKind kind, uint32_t binding) {
    switch (kind) {
        case DescriptorKind::Constant:
            return binding >= kConstantBindingBase && binding < kConstantBindingBase + kConstantBindingCount;
        case DescriptorKind::Texture:
            return binding >= kTextureBindingBase && binding < kTextureBindingBase + kTextureBindingCount;
        case DescriptorKind::Sampler:
            return binding >= kSamplerBindingBase && binding < kSamplerBindingBase + kSamplerBindingCount;
        default: return false;
    }
}

const char* DescriptorKindName(DescriptorKind kind) {
    switch (kind) {
        case DescriptorKind::Constant: return "constant";
        case DescriptorKind::Texture: return "texture";
        case DescriptorKind::Sampler: return "sampler";
        default: return "unsupported";
    }
}

bool DecodeReflectedStage(const std::vector<uint32_t>& spirv, ShaderReflection& reflection, std::string& error) {
    if (spirv.size() < 5 || spirv[0] != kSpirvMagic) {
        error = "SPIR-V module has an invalid header";
        return false;
    }

    std::unordered_map<uint32_t, std::string> names;
    std::unordered_map<uint32_t, Decorations> decorations;
    std::unordered_map<uint32_t, TypeInfo> types;
    std::vector<Variable> variables;
    std::vector<MemberOffset> member_offsets;
    bool found_entry = false;
    std::string entry;
    ShaderStage stage = ShaderStage::Vertex;

    for (size_t cursor = 5; cursor < spirv.size();) {
        const uint32_t word = spirv[cursor];
        const uint32_t words = word >> 16;
        const uint16_t opcode = static_cast<uint16_t>(word & 0xffffu);
        if (words == 0 || cursor + words > spirv.size()) {
            error = "SPIR-V instruction has an invalid length";
            return false;
        }
        const uint32_t* operands = spirv.data() + cursor;
        switch (opcode) {
            case kOpName:
                if (words >= 3) names[operands[1]] = SpvString(operands, words, 2);
                break;
            case kOpEntryPoint:
                if (words < 4) {
                    error = "SPIR-V entry point is malformed";
                    return false;
                }
                if (found_entry) {
                    error = "SPIR-V module contains multiple entry points";
                    return false;
                }
                if (!DecodeStage(operands[1], stage)) {
                    error = "SPIR-V entry point uses an unsupported execution model";
                    return false;
                }
                entry = SpvString(operands, words, 3);
                found_entry = true;
                break;
            case kOpTypeStruct:
                if (words >= 2) {
                    TypeInfo& type = types[operands[1]];
                    type.opcode = opcode;
                    type.members.assign(operands + 2, operands + words);
                }
                break;
            case kOpTypePointer:
                if (words >= 4) {
                    TypeInfo& type = types[operands[1]];
                    type.opcode = opcode;
                    type.pointee = operands[3];
                }
                break;
            case kOpTypeVoid:
            case kOpTypeBool:
            case kOpTypeInt:
            case kOpTypeFloat:
            case kOpTypeVector:
            case kOpTypeMatrix:
            case kOpTypeArray:
            case kOpTypeRuntimeArray:
                if (words >= 3) {
                    types[operands[1]].opcode = opcode;
                    types[operands[1]].pointee = operands[2];
                }
                break;
            case kOpTypeImage:
                if (words >= 8) {
                    TypeInfo& type = types[operands[1]];
                    type.opcode = opcode;
                    type.has_sampled_operand = true;
                    type.sampled_operand = operands[7];
                }
                break;
            case kOpTypeSampler:
            case kOpTypeSampledImage:
                if (words >= 2) types[operands[1]].opcode = opcode;
                break;
            case kOpVariable:
                if (words >= 4) variables.push_back({operands[1], operands[2], operands[3]});
                break;
            case kOpDecorate:
                if (words >= 3) {
                    Decorations& decoration = decorations[operands[1]];
                    if (operands[2] == kDecorationBinding && words >= 4) {
                        decoration.has_binding = true;
                        decoration.binding = operands[3];
                    } else if (operands[2] == kDecorationDescriptorSet && words >= 4) {
                        decoration.has_set = true;
                        decoration.set = operands[3];
                    } else if (operands[2] == kDecorationLocation && words >= 4) {
                        decoration.has_location = true;
                        decoration.location = operands[3];
                    } else if (operands[2] == kDecorationBlock) {
                        decoration.has_block = true;
                    }
                }
                break;
            case kOpMemberDecorate:
                if (words >= 5 && operands[3] == kDecorationOffset)
                    member_offsets.push_back({operands[1], operands[2], operands[4]});
                break;
            default: break;
        }
        cursor += words;
    }

    if (!found_entry) {
        error = "SPIR-V module has no entry point";
        return false;
    }
    reflection = {};
    reflection.stage = stage;
    reflection.entry = entry;
    std::unordered_set<uint64_t> seen_bindings;
    for (const Variable& variable : variables) {
        const auto found = decorations.find(variable.id);
        if (found == decorations.end()) continue;
        const Decorations& decoration = found->second;
        if (decoration.has_binding) {
            if (!decoration.has_set || decoration.set != 0) {
                error = "shader ABI v1: descriptor is outside set 0";
                return false;
            }
            const auto type_it = types.find(variable.type);
            if (type_it == types.end() || type_it->second.opcode != kOpTypePointer) {
                error = "shader ABI v1: descriptor pointer type is missing";
                return false;
            }
            const DescriptorKind kind = DescriptorType(type_it->second.pointee, types, decorations);
            const bool storage_matches =
                (kind == DescriptorKind::Constant && variable.storage == kStorageUniform) ||
                ((kind == DescriptorKind::Texture || kind == DescriptorKind::Sampler) &&
                 variable.storage == kStorageUniformConstant);
            if (!storage_matches) {
                error = "shader ABI v1: descriptor storage class does not match its reflected type";
                return false;
            }
            if (!BindingInRange(kind, decoration.binding)) {
                error = std::string("shader ABI v1: ") + DescriptorKindName(kind) + " binding " +
                        std::to_string(decoration.binding) + " is outside its typed range";
                return false;
            }
            const uint64_t key = (static_cast<uint64_t>(decoration.set) << 32) | decoration.binding;
            if (!seen_bindings.insert(key).second) {
                error = "shader ABI v1: duplicate reflected descriptor set/binding";
                return false;
            }
            ShaderBinding::Kind reflected_kind = ShaderBinding::Kind::Constant;
            if (kind == DescriptorKind::Texture) reflected_kind = ShaderBinding::Kind::Texture;
            if (kind == DescriptorKind::Sampler) reflected_kind = ShaderBinding::Kind::Sampler;
            reflection.bindings.push_back({0, decoration.binding, reflected_kind, names[variable.id]});
        }
        if (decoration.has_location &&
            (variable.storage == kStorageInput || variable.storage == kStorageOutput)) {
            reflection.locations.push_back(
                {decoration.location, variable.storage == kStorageInput, names[variable.id]});
        }
        if (variable.storage == kStorageUniform) {
            const auto type_it = types.find(variable.type);
            if (type_it != types.end() && type_it->second.opcode == kOpTypePointer) {
                const auto struct_it = types.find(type_it->second.pointee);
                if (struct_it != types.end() && struct_it->second.opcode == kOpTypeStruct) {
                    for (const MemberOffset& member : member_offsets)
                        if (member.type == type_it->second.pointee && decoration.has_binding)
                            reflection.uniform_member_offsets.push_back(
                                {decoration.binding, member.member, member.offset, names[variable.id]});
                }
            }
        }
    }
    std::sort(reflection.bindings.begin(), reflection.bindings.end(),
              [](const ShaderBinding& a, const ShaderBinding& b) { return a.binding < b.binding; });
    std::sort(reflection.locations.begin(), reflection.locations.end(),
              [](const ShaderLocation& a, const ShaderLocation& b) {
                  if (a.input != b.input) return a.input < b.input;
                  return a.location < b.location;
              });
    std::sort(reflection.uniform_member_offsets.begin(), reflection.uniform_member_offsets.end(),
              [](const UniformMemberOffset& a, const UniformMemberOffset& b) {
                  if (a.binding != b.binding) return a.binding < b.binding;
                  return a.member < b.member;
              });
    return true;
}

std::string AdaptHlsl(const std::string& source) {
    std::string adapted = source;
    for (size_t at; (at = adapted.find("1.#INF")) != std::string::npos;)
        adapted.replace(at, 6, "asfloat(0x7F800000u)");
    static const std::regex attr(R"((\n[ \t]*)(\w+ v\d+ : ATTR(\d+);))");
    return std::regex_replace(adapted, attr, "$1[[vk::location($3)]] $2");
}

}  // namespace

bool ReflectShader(const std::vector<uint32_t>& spirv, ShaderReflection& reflection, std::string& error) {
    reflection = {};
    error.clear();
    ShaderReflection candidate;
    if (!DecodeReflectedStage(spirv, candidate, error)) return false;
    reflection = std::move(candidate);
    return true;
}

bool CompileShader(const std::string& hlsl, const char* entry, ShaderStage stage, ShaderModule& module,
                   std::string& error) {
    module = {};
    error.clear();
    if (!entry || !*entry) {
        error = "shader compiler requires a non-empty entry point";
        return false;
    }
    static std::once_flag init;
    static bool initialized = false;
    std::call_once(init, [] { initialized = glslang::InitializeProcess(); });
    if (!initialized) {
        error = "glslang initialization failed";
        return false;
    }
    const EShLanguage language = Language(stage);
    const std::string source = AdaptHlsl(hlsl);
    const char* text = source.c_str();
    const int length = static_cast<int>(source.size());
    glslang::TShader shader(language);
    shader.setStringsWithLengths(&text, &length, 1);
    shader.setEntryPoint(entry);
    shader.setSourceEntryPoint(entry);
    shader.setEnvInput(glslang::EShSourceHlsl, language, glslang::EShClientVulkan, 100);
    shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_3);
    shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_6);
    shader.setHlslIoMapping(true);
    shader.setAutoMapBindings(false);
    shader.setAutoMapLocations(true);
    shader.setShiftBinding(glslang::EResUbo, kConstantBindingBase);
    shader.setShiftBinding(glslang::EResTexture, kTextureBindingBase);
    shader.setShiftBinding(glslang::EResSampler, kSamplerBindingBase);
    const auto messages = EShMessages(EShMsgSpvRules | EShMsgVulkanRules | EShMsgReadHlsl | EShMsgHlslOffsets);
    if (!shader.parse(GetDefaultResources(), 100, false, messages)) {
        error = std::string("glslang ") + StageName(stage) + " parse failed: " + shader.getInfoLog();
        return false;
    }
    glslang::TProgram program;
    program.addShader(&shader);
    if (!program.link(messages) || !program.mapIO()) {
        error = std::string("glslang ") + StageName(stage) + " link failed: " + program.getInfoLog();
        return false;
    }
    glslang::SpvOptions options;
    options.disableOptimizer = true;
    options.validate = false;
    ShaderModule candidate;
    glslang::GlslangToSpv(*program.getIntermediate(language), candidate.spirv, &options);
    if (candidate.spirv.empty()) {
        error = std::string("glslang ") + StageName(stage) + " emitted an empty SPIR-V module";
        return false;
    }
    if (!ReflectShader(candidate.spirv, candidate.reflection, error)) return false;
    if (candidate.reflection.stage != stage || candidate.reflection.entry != entry) {
        error = std::string("SPIR-V reflection disagrees with requested ") + StageName(stage) + " entry point";
        return false;
    }
    module = std::move(candidate);
    return true;
}

}  // namespace gpu::vulkan
