#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace gpu::vulkan {

// Metadata shared by the backend ring and the focused completion fixture. The owning slot keeps the Vulkan
// readback buffer separately; this state is the only part consulted for reuse/delivery eligibility.
struct LateReadbackSlotState {
    uint64_t frame_generation = 0;
    uint64_t identity_generation = 0;
    uint64_t submission_serial = 0;
    uint64_t delivered_serial = 0;
    bool valid = false;
};

inline bool LateNeedsWrite(const LateReadbackSlotState& slot, uint64_t delivered_serial) {
    return slot.submission_serial > delivered_serial;
}

template <size_t N>
bool LateSlotsRetired(const std::array<LateReadbackSlotState, N>& slots, uint64_t completed_serial) {
    for (const auto& slot : slots)
        if (slot.valid && (!slot.submission_serial || slot.submission_serial > completed_serial)) return false;
    return true;
}

template <size_t N>
int LateRecordSlot(const std::array<LateReadbackSlotState, N>& slots, size_t next, uint64_t completed_serial) {
    if (next >= N) return -1;
    const auto& slot = slots[next];
    if (slot.valid && (!slot.submission_serial || slot.submission_serial > completed_serial)) return -1;
    return static_cast<int>(next);
}

template <size_t N>
int LateDeliverySlot(const std::array<LateReadbackSlotState, N>& slots, uint64_t completed_serial,
                     uint64_t published_generation, uint64_t current_identity, uint64_t recent_window) {
    int best = -1;
    for (size_t i = 0; i < N; ++i) {
        const auto& slot = slots[i];
        const bool recent = published_generation >= slot.frame_generation &&
                            published_generation - slot.frame_generation <= recent_window;
        if (!slot.valid || !slot.submission_serial || slot.submission_serial > completed_serial ||
            slot.identity_generation != current_identity || !recent)
            continue;
        if (best < 0 || slot.submission_serial > slots[static_cast<size_t>(best)].submission_serial)
            best = static_cast<int>(i);
    }
    return best;
}

}  // namespace gpu::vulkan
