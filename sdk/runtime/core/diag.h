#pragma once
#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <optional>
#include <string>
#include <string_view>

// diagnostics switched on by environment variables, read once
namespace diag {

// XBE_PERF set: perf log lines once a second and the read-backs named
inline bool Perf() {
    static const bool on = std::getenv("XBE_PERF") != nullptr;
    return on;
}

// XBE_CAPTURE_DIR: where F3 saves a frame, empty: F3 does not capture
inline const std::filesystem::path& CaptureDir() {
    static const std::filesystem::path dir = [] {
#ifdef _WIN32
        const wchar_t* v = _wgetenv(L"XBE_CAPTURE_DIR");
#else
        const char* v = std::getenv("XBE_CAPTURE_DIR");
#endif
        return v ? std::filesystem::path(v) : std::filesystem::path();
    }();
    return dir;
}

// XBE_PERF_TRACE: a bounded, per-frame CSV trace for an explicitly requested run. It is
// deliberately separate from the once-a-second aggregate diagnostics so percentile analysis
// cannot accidentally treat an aggregate window as a set of individual samples.
struct PerfFrameSample {
    uint64_t index = 0;
    double renderer_sample_ms = 0.0;
    std::optional<double> record_ms;
    std::optional<double> wait_ms;
    std::optional<double> gpu_ms;
    std::optional<double> compile_ms;
    std::optional<double> upload_ms;
    std::optional<double> sync_ms;
    std::optional<double> present_ms;
    std::optional<uint64_t> draws;
    std::optional<uint64_t> shader_events;
    std::optional<uint64_t> pipeline_events;
    std::optional<uint64_t> texture_uploads;
    std::optional<uint64_t> auxiliary_events;
    std::optional<uint64_t> source_hits;
    std::optional<uint64_t> source_misses;
    std::optional<double> detail_vertex_assembly_ms;
    std::optional<double> detail_texture_target_sampler_ms;
    std::optional<double> detail_pipeline_source_ms;
    std::optional<double> detail_constant_vertex_index_upload_ms;
    std::optional<double> detail_descriptor_lookup_allocation_ms;
    std::optional<double> detail_command_recording_ms;
    std::optional<double> detail_phase_sum_ms;
    std::optional<double> detail_record_residual_ms;
    std::optional<double> detail_swapchain_render_wait_ms;
    std::optional<double> detail_batch_completion_wait_ms;
    std::optional<double> detail_late_publish_ms;
    std::optional<double> detail_descriptor_retire_ms;
    std::optional<double> detail_resource_retire_ms;
    std::optional<double> detail_wait_phase_sum_ms;
    std::optional<double> detail_wait_residual_ms;
};

class PerfTrace {
public:
    explicit PerfTrace(std::string_view backend) : backend_(backend) {}

    static bool Requested() { return !TracePath().empty(); }
    bool Enabled() {
        Open();
        return file_.is_open() && !closed_;
    }

    void Frame(const PerfFrameSample& sample) {
        if (!Enabled()) return;
        if (!StartIfNeeded()) return;
        if (frame_rows_ >= max_frames_ || rows_ >= max_rows_) return CloseAtLimit();
        bool first = true;
        Column(first, "frame");
        Column(first, backend_);
        Column(first, sample.index);
        Column(first, "");
        Column(first, sample.renderer_sample_ms);
        Column(first, std::optional<double>());
        Column(first, sample.record_ms);
        Column(first, sample.wait_ms);
        Column(first, sample.gpu_ms);
        Column(first, sample.compile_ms);
        Column(first, sample.upload_ms);
        Column(first, sample.sync_ms);
        Column(first, sample.present_ms);
        Column(first, sample.draws);
        Column(first, sample.shader_events);
        Column(first, sample.pipeline_events);
        Column(first, sample.texture_uploads);
        Column(first, sample.auxiliary_events);
        Column(first, sample.source_hits);
        Column(first, sample.source_misses);
        Column(first, sample.gpu_ms.has_value() ? 1u : 0u);
        Column(first, sample.detail_vertex_assembly_ms);
        Column(first, sample.detail_texture_target_sampler_ms);
        Column(first, sample.detail_pipeline_source_ms);
        Column(first, sample.detail_constant_vertex_index_upload_ms);
        Column(first, sample.detail_descriptor_lookup_allocation_ms);
        Column(first, sample.detail_command_recording_ms);
        Column(first, sample.detail_phase_sum_ms);
        Column(first, sample.detail_record_residual_ms);
        Column(first, sample.detail_swapchain_render_wait_ms);
        Column(first, sample.detail_batch_completion_wait_ms);
        Column(first, sample.detail_late_publish_ms);
        Column(first, sample.detail_descriptor_retire_ms);
        Column(first, sample.detail_resource_retire_ms);
        Column(first, sample.detail_wait_phase_sum_ms);
        Column(first, sample.detail_wait_residual_ms);
        file_ << '\n';
        ++frame_rows_;
        ++rows_;
        FlushPeriodically();
    }

    void Gpu(uint64_t index, double gpu_ms) {
        if (!Enabled()) return;
        if (!StartIfNeeded()) return;
        if (rows_ >= max_rows_) return CloseAtLimit();
        bool first = true;
        Column(first, "gpu");
        Column(first, backend_);
        Column(first, index);
        Column(first, "");
        Column(first, std::optional<double>());
        Column(first, std::optional<double>());
        Column(first, std::optional<double>());
        Column(first, std::optional<double>());
        Column(first, gpu_ms);
        Column(first, std::optional<double>());
        Column(first, std::optional<double>());
        Column(first, std::optional<double>());
        Column(first, std::optional<double>());
        Column(first, std::optional<uint64_t>());
        Column(first, std::optional<uint64_t>());
        Column(first, std::optional<uint64_t>());
        Column(first, std::optional<uint64_t>());
        Column(first, std::optional<uint64_t>());
        Column(first, std::optional<uint64_t>());
        Column(first, std::optional<uint64_t>());
        Column(first, 1u);
        for (int i = 0; i < 15; ++i) Column(first, std::optional<double>());
        file_ << '\n';
        ++rows_;
        FlushPeriodically();
    }

    // Records the interval between renderer Present entry points. This is a shared CPU
    // boundary, not display scanout, guest simulation time, or a GPU timestamp.
    void Present(bool armed) {
        if (!Requested() || !armed || !Enabled()) return;
        if (!StartIfNeeded()) return;
        const auto now = std::chrono::steady_clock::now();
        if (!have_present_) {
            last_present_ = now;
            have_present_ = true;
            return;
        }
        if (present_rows_ >= max_frames_ || rows_ >= max_rows_) return CloseAtLimit();
        bool first = true;
        Column(first, "present");
        Column(first, backend_);
        Column(first, present_index_++);
        Column(first, "");
        Column(first, std::optional<double>());
        Column(first, std::chrono::duration<double, std::milli>(now - last_present_).count());
        for (int i = 0; i < 30; i++) Column(first, std::optional<double>());
        file_ << '\n';
        last_present_ = now;
        ++present_rows_;
        ++rows_;
        FlushPeriodically();
    }

    // The control command owns the measurement boundary. This keeps startup and menu Presents
    // out of the finite trace and leaves an auditable end marker when recording stops.
    void Stop() {
        if (!started_ || closed_ || !file_.is_open()) return;
        if (rows_ < max_rows_) Marker("measure_stop");
        file_.flush();
        closed_ = true;
    }

private:
    static const std::filesystem::path& TracePath() {
        static const std::filesystem::path path = [] {
            const char* value = std::getenv("XBE_PERF_TRACE");
            return value && value[0] ? std::filesystem::path(value) : std::filesystem::path();
        }();
        return path;
    }

    void Open() {
        if (opened_) return;
        opened_ = true;
        if (TracePath().empty()) return;
        const auto limit = [](const char* name, uint64_t fallback) {
            const char* value = std::getenv(name);
            if (!value || !value[0]) return fallback;
            char* end = nullptr;
            if (value[0] == '-' || value[0] == '+') return fallback;
            errno = 0;
            const unsigned long long parsed = std::strtoull(value, &end, 10);
            if (end == value || !end || *end != '\0' || errno == ERANGE || parsed == 0) return fallback;
            return uint64_t(std::min<unsigned long long>(parsed, 1000000ull));
        };
        max_rows_ = limit("XBE_PERF_TRACE_MAX_ROWS", max_rows_);
        max_frames_ = limit("XBE_PERF_TRACE_MAX_FRAMES", max_frames_);
        file_.open(TracePath(), std::ios::out | std::ios::trunc);
        if (!file_) {
            std::fprintf(stderr, "diag: XBE_PERF_TRACE requested but the trace file could not be opened\n");
            closed_ = true;
            return;
        }
        file_ << "codf_perf_trace_v1\n"
                 "kind,backend,sample_index,label,renderer_sample_ms,renderer_present_interval_ms,record_ms,wait_ms,gpu_ms,"
                 "compile_ms,upload_ms,sync_ms,present_ms,draws,shader_events,pipeline_events,texture_uploads,"
                 "auxiliary_events,source_hits,source_misses,gpu_valid,detail_vertex_assembly_ms,"
                 "detail_texture_target_sampler_ms,detail_pipeline_source_ms,detail_constant_vertex_index_upload_ms,"
                 "detail_descriptor_lookup_allocation_ms,detail_command_recording_ms,detail_phase_sum_ms,"
                 "detail_record_residual_ms,detail_swapchain_render_wait_ms,detail_batch_completion_wait_ms,"
                 "detail_late_publish_ms,detail_descriptor_retire_ms,detail_resource_retire_ms,"
                 "detail_wait_phase_sum_ms,detail_wait_residual_ms\n";
        file_ << std::fixed << std::setprecision(6);
    }

    template <typename T>
    void Column(bool& first, const T& value) {
        if (!first) file_ << ',';
        file_ << value;
        first = false;
    }

    template <typename T>
    void Column(bool& first, const std::optional<T>& value) {
        if (!first) file_ << ',';
        if (value) file_ << *value;
        first = false;
    }

    void FlushPeriodically() {
        if (rows_ % 64 == 0) file_.flush();
    }

    bool StartIfNeeded() {
        if (started_) return !closed_;
        if (rows_ >= max_rows_) {
            CloseAtLimit();
            return false;
        }
        Marker("measure_start");
        started_ = true;
        return true;
    }

    void Marker(std::string_view label) {
        bool first = true;
        Column(first, "marker");
        Column(first, backend_);
        Column(first, uint64_t(0));
        Column(first, label);
        for (int i = 0; i < 32; ++i) Column(first, std::optional<double>());
        file_ << '\n';
        ++rows_;
        FlushPeriodically();
    }

    void CloseAtLimit() {
        file_.flush();
        closed_ = true;
    }

    std::string backend_;
    std::ofstream file_;
    uint64_t rows_ = 0;
    uint64_t frame_rows_ = 0;
    uint64_t present_rows_ = 0;
    uint64_t present_index_ = 0;
    uint64_t max_rows_ = 8192;
    uint64_t max_frames_ = 4096;
    bool opened_ = false;
    bool closed_ = false;
    bool started_ = false;
    bool have_present_ = false;
    std::chrono::steady_clock::time_point last_present_;
};

}  // namespace diag
