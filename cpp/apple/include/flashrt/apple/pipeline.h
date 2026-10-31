#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace flashrt::apple {

namespace detail { struct Context; struct Buffer; struct Kernel; struct Pipeline; }

// Synchronous, shared-memory native Metal runtime. Copies share ownership.
// All uploads, downloads and launches on one context are serialized, including
// GPU completion. A host mapping is valid only while its Buffer owner lives;
// callers must not access it concurrently with a launch or another writer.
class Buffer {
public:
    std::size_t size() const;
    void write(const void* source, std::size_t bytes, std::size_t offset = 0) const;
    void read(void* destination, std::size_t bytes, std::size_t offset = 0) const;
    void* host_data() const;
private:
    explicit Buffer(std::shared_ptr<detail::Buffer> impl);
    std::shared_ptr<detail::Buffer> impl_;
    friend class Context;
};

class Kernel {
private:
    explicit Kernel(std::shared_ptr<detail::Kernel> impl);
    std::shared_ptr<detail::Kernel> impl_;
    friend class Context;
};

enum class Access { read, write, read_write };
struct Binding {
    Buffer buffer;
    std::size_t offset = 0;
    Access access = Access::read;
};
struct Size { std::size_t x = 1, y = 1, z = 1; };
struct Dispatch {
    Kernel kernel;
    // Position in this vector is the Metal buffer argument index (0..30).
    std::vector<Binding> bindings;
    Size threadgroups;
    Size threads_per_threadgroup;
    std::string label;
};
struct BufferCopy {
    Binding source;
    Binding destination;
    std::size_t bytes;
    std::string label;
};
using Operation = std::variant<Dispatch, BufferCopy>;
enum class Execution { direct, icb };
struct Timing {
    double encode_us = 0;
    double gpu_us = 0;
    double wall_us = 0;
    double commit_us = 0;
    double wait_us = 0;
    double commit_to_gpu_us = 0;
    double gpu_to_return_us = 0;
    double kernel_us = 0;
    double submit_host_s = 0;
    double gpu_start_s = 0;
    double gpu_end_s = 0;
    double return_host_s = 0;
};

class Pipeline {
public:
    // Immutable bindings and geometry; buffer contents may change between runs.
    // Sequential dependencies are enforced between operations. Buffer copies
    // use blit encoders; compute spans reuse ranges of the recorded ICB.
    Timing run(Execution mode = Execution::direct) const;
    // Keep this fixed plan's resources resident between submissions (macOS 15+).
    void set_resident(bool enabled) const;
    std::size_t dispatch_count() const;
private:
    explicit Pipeline(std::shared_ptr<detail::Pipeline> impl);
    std::shared_ptr<detail::Pipeline> impl_;
    friend class Context;
};

class Context {
public:
    Context();
    std::string device_name() const;
    Buffer buffer(std::size_t bytes) const;
    // Runtime MSL compilation: no offline metal compiler or Python dependency.
    Kernel kernel(const std::string& source, const std::string& entry_point,
                  bool tensor_ops = false) const;
    // Owns all referenced resources; ICB is recorded once at construction.
    Pipeline pipeline(const std::vector<Dispatch>& dispatches) const;
    Pipeline program(const std::vector<Operation>& operations,
                     const std::string& label = "FlashRT Apple") const;
private:
    std::shared_ptr<detail::Context> impl_;
};

} // namespace flashrt::apple
