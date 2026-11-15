#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <flashrt/apple/pipeline.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <mutex>
#include <map>
#include <stdexcept>
#include <utility>
#include <mach/mach_time.h>

namespace flashrt::apple {
namespace detail {
struct Context {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    std::mutex mutex;
};
struct Buffer { std::shared_ptr<Context> context; id<MTLBuffer> value; };
struct Kernel { std::shared_ptr<Context> context; id<MTLComputePipelineState> value; };
struct Node {
    NSString* label = nil;
    std::shared_ptr<Kernel> kernel;
    std::vector<std::shared_ptr<Buffer>> buffers;
    std::vector<std::size_t> offsets;
    MTLSize groups{}, threads{};
    std::size_t copy_bytes = 0;
    int external = 0;               // 1: ExternalSignal, 2: ExternalWait
    std::uint32_t task = 0;
};
enum class BatchKind { compute, copy, signal, wait };
struct Batch { std::size_t begin, count, icb_begin; BatchKind kind; };
struct Pipeline {
    NSString* label = nil;
    std::shared_ptr<Context> context;
    std::vector<Node> nodes;
    std::vector<Batch> batches;
    std::size_t dispatch_count = 0;
    std::map<id<MTLBuffer>, MTLResourceUsage> resources;
    id<MTLIndirectCommandBuffer> icb = nil;
    id<MTLResidencySet> residency API_AVAILABLE(macos(15.0)) = nil;
    // External tasks: one shared event; per run, task t is signalled at base+2t+1 and
    // released at base+2t+2. Bases grow monotonically across runs.
    id<MTLSharedEvent> event = nil;
    std::vector<std::uint32_t> signal_order;
    std::uint64_t event_base = 0;
    ~Pipeline() {
        if (@available(macOS 15.0, *)) {
            if (residency) {
                [context->queue removeResidencySet:residency];
                [residency endResidency];
            }
        }
    }
};
}
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::micro>(end - start).count();
}
double host_seconds() {
    static const double scale = [] {
        mach_timebase_info_data_t info;
        mach_timebase_info(&info);
        return double(info.numer) / double(info.denom) * 1e-9;
    }();
    return double(mach_absolute_time()) * scale;
}
void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}
std::runtime_error failure(const char* operation, NSError* error) {
    return std::runtime_error(std::string(operation) + ": " +
        (error ? error.localizedDescription.UTF8String : "Metal returned nil"));
}
void range(std::size_t size, std::size_t offset, std::size_t bytes) {
    require(offset <= size && bytes <= size - offset, "buffer range out of bounds");
}
MTLSize dimensions(Size s) {
    require(s.x && s.y && s.z, "dispatch dimensions must be positive");
    // Metal dispatch dimensions are represented by 32-bit shader coordinates.
    require(s.x <= UINT32_MAX && s.y <= UINT32_MAX && s.z <= UINT32_MAX,
            "dispatch dimensions exceed uint32");
    return MTLSizeMake(s.x, s.y, s.z);
}
}

Buffer::Buffer(std::shared_ptr<detail::Buffer> impl) : impl_(std::move(impl)) {}
Kernel::Kernel(std::shared_ptr<detail::Kernel> impl) : impl_(std::move(impl)) {}
Pipeline::Pipeline(std::shared_ptr<detail::Pipeline> impl) : impl_(std::move(impl)) {}
void Pipeline::set_resident(bool enabled) const {
    std::lock_guard<std::mutex> lock(impl_->context->mutex);
    @autoreleasepool {
        if (@available(macOS 15.0, *)) {
            if (enabled == (impl_->residency != nil)) return;
            if (!enabled) {
                [impl_->context->queue removeResidencySet:impl_->residency];
                [impl_->residency endResidency];
                impl_->residency = nil;
                return;
            }
            MTLResidencySetDescriptor* descriptor = [MTLResidencySetDescriptor new];
            descriptor.label = impl_->label;
            descriptor.initialCapacity = impl_->resources.size() + 1;
            NSError* error = nil;
            id<MTLResidencySet> set = [impl_->context->device newResidencySetWithDescriptor:descriptor error:&error];
            if (!set) throw failure("create residency set", error);
            for (const auto& resource : impl_->resources) [set addAllocation:resource.first];
            if (impl_->icb) [set addAllocation:impl_->icb];
            [set commit];
            [set requestResidency];
            [impl_->context->queue addResidencySet:set];
            impl_->residency = set;
        } else if (enabled) {
            throw std::runtime_error("Explicit residency requires macOS 15 or newer");
        }
    }
}
std::size_t Buffer::size() const { return impl_->value.length; }
void* Buffer::host_data() const { return impl_->value.contents; }
void Buffer::write(const void* source, std::size_t bytes, std::size_t offset) const {
    range(size(), offset, bytes);
    require(source || !bytes, "null upload source");
    std::lock_guard<std::mutex> lock(impl_->context->mutex);
    if (bytes) std::memcpy(static_cast<char*>(impl_->value.contents) + offset, source, bytes);
}
void Buffer::read(void* destination, std::size_t bytes, std::size_t offset) const {
    range(size(), offset, bytes);
    require(destination || !bytes, "null download destination");
    std::lock_guard<std::mutex> lock(impl_->context->mutex);
    if (bytes) std::memcpy(destination, static_cast<char*>(impl_->value.contents) + offset, bytes);
}
Context::Context() : impl_(std::make_shared<detail::Context>()) {
    @autoreleasepool {
        impl_->device = MTLCreateSystemDefaultDevice();
        if (!impl_->device) throw std::runtime_error("No Metal device available");
        if (!impl_->device.hasUnifiedMemory)
            throw std::runtime_error("Apple native pipeline requires unified memory");
        impl_->queue = [impl_->device newCommandQueue];
        if (!impl_->queue) throw failure("create command queue", nil);
        impl_->queue.label = @"FlashRT Apple native";
    }
}
std::string Context::device_name() const { return impl_->device.name.UTF8String; }
Buffer Context::buffer(std::size_t bytes) const {
    require(bytes && bytes <= impl_->device.maxBufferLength, "invalid buffer size");
    @autoreleasepool {
        auto b = std::make_shared<detail::Buffer>();
        b->context = impl_;
        b->value = [impl_->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!b->value) throw failure("allocate shared buffer", nil);
        return Buffer(std::move(b));
    }
}
Kernel Context::kernel(const std::string& source, const std::string& entry_point, bool tensor_ops) const {
    @autoreleasepool {
        NSError* error = nil;
        NSString* code = [[NSString alloc] initWithBytes:source.data() length:source.size()
                                               encoding:NSUTF8StringEncoding];
        NSString* name = [[NSString alloc] initWithBytes:entry_point.data() length:entry_point.size()
                                               encoding:NSUTF8StringEncoding];
        require(code && name, "MSL source and entry point must be UTF-8");
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.fastMathEnabled = NO;
        if (tensor_ops) {
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
            if (@available(macOS 26.2, *)) options.languageVersion = MTLLanguageVersion4_0;
            else throw std::runtime_error("BF16 TensorOps requires macOS 26.2 or newer");
#else
            throw std::runtime_error("BF16 TensorOps requires a macOS 26+ SDK build");
#endif
        }
        id<MTLLibrary> library = [impl_->device newLibraryWithSource:code options:options error:&error];
        if (!library) throw failure(("compile MSL ("+entry_point+")").c_str(), error);
        id<MTLFunction> function = [library newFunctionWithName:name];
        require(function != nil, "MSL entry point not found");
        MTLComputePipelineDescriptor* descriptor = [MTLComputePipelineDescriptor new];
        descriptor.computeFunction = function;
        descriptor.label = name;
        descriptor.supportIndirectCommandBuffers = YES;
        auto k = std::make_shared<detail::Kernel>();
        k->context = impl_;
        k->value = [impl_->device newComputePipelineStateWithDescriptor:descriptor
            options:MTLPipelineOptionNone reflection:nil error:&error];
        if (!k->value) throw failure("create compute pipeline", error);
        require(k->value.supportIndirectCommandBuffers, "kernel does not support ICB");
        if (tensor_ops) require(k->value.threadExecutionWidth == 32,
                               "BF16 kernels require 32-thread SIMD groups");
        return Kernel(std::move(k));
    }
}
Pipeline Context::pipeline(const std::vector<Dispatch>& dispatches) const {
    return program(std::vector<Operation>(dispatches.begin(), dispatches.end()));
}
Pipeline Context::program(const std::vector<Operation>& operations, const std::string& label) const {
    require(!operations.empty(), "pipeline must contain at least one operation");
    require(operations.size() <= UINT32_MAX, "too many operations");
    @autoreleasepool {
        auto p = std::make_shared<detail::Pipeline>();
        p->context = impl_;
        p->label = [NSString stringWithUTF8String:label.c_str()];
        std::size_t max_bindings = 0;
        std::vector<int> task_state;   // 0 unseen, 1 signalled, 2 waited
        for (const auto& operation : operations) {
            detail::Node node;
            const ExternalSignal* signal = std::get_if<ExternalSignal>(&operation);
            const ExternalWait* wait = std::get_if<ExternalWait>(&operation);
            if (signal || wait) {
                const std::uint32_t task = signal ? signal->task : wait->task;
                if (task_state.size() <= task) task_state.resize(task + 1, 0);
                require(signal ? task_state[task] == 0 : task_state[task] == 1,
                        "external task needs one signal followed by one wait");
                task_state[task] = signal ? 1 : 2;
                if (signal) p->signal_order.push_back(task);
                node.label = [NSString stringWithUTF8String:(signal ? signal->label : wait->label).c_str()];
                node.external = signal ? 1 : 2;
                node.task = task;
                p->batches.push_back({p->nodes.size(), 1, p->dispatch_count,
                                      signal ? detail::BatchKind::signal : detail::BatchKind::wait});
                p->nodes.push_back(std::move(node));
                continue;
            }
            if (const auto* copy = std::get_if<BufferCopy>(&operation)) {
                require(copy->bytes > 0, "buffer copy must be nonempty");
                const auto& src = copy->source;
                const auto& dst = copy->destination;
                require(src.buffer.impl_->context == impl_ && dst.buffer.impl_->context == impl_,
                        "copy buffer belongs to another context");
                range(src.buffer.size(), src.offset, copy->bytes);
                range(dst.buffer.size(), dst.offset, copy->bytes);
                require(src.buffer.impl_ != dst.buffer.impl_ ||
                        src.offset + copy->bytes <= dst.offset ||
                        dst.offset + copy->bytes <= src.offset, "buffer copy ranges overlap");
                node.label = [NSString stringWithUTF8String:copy->label.c_str()];
                node.copy_bytes = copy->bytes;
                node.buffers = {src.buffer.impl_, dst.buffer.impl_};
                node.offsets = {src.offset, dst.offset};
                p->resources[src.buffer.impl_->value] |= MTLResourceUsageRead;
                p->resources[dst.buffer.impl_->value] |= MTLResourceUsageWrite;
            } else {
                const auto& d = std::get<Dispatch>(operation);
                require(d.kernel.impl_->context == impl_, "kernel belongs to another context");
                require(d.bindings.size() <= 31, "Metal supports at most 31 buffer arguments");
                node.label = [NSString stringWithUTF8String:d.label.c_str()];
                node.kernel = d.kernel.impl_;
                node.groups = dimensions(d.threadgroups);
                node.threads = dimensions(d.threads_per_threadgroup);
                auto limit = node.kernel->value.maxTotalThreadsPerThreadgroup;
                auto axis = impl_->device.maxThreadsPerThreadgroup;
                require(node.threads.width <= axis.width && node.threads.height <= axis.height &&
                        node.threads.depth <= axis.depth, "threadgroup axis exceeds device limit");
                require(node.threads.width <= limit / node.threads.height &&
                        node.threads.width * node.threads.height <= limit / node.threads.depth,
                        "threadgroup exceeds kernel limit");
                for (const auto& b : d.bindings) {
                    require(b.buffer.impl_->context == impl_, "buffer belongs to another context");
                    require(b.offset < b.buffer.size() && b.offset % 4 == 0,
                            "buffer binding offset must be in bounds and 4-byte aligned");
                    MTLResourceUsage usage;
                    switch (b.access) {
                        case Access::read: usage = MTLResourceUsageRead; break;
                        case Access::write: usage = MTLResourceUsageWrite; break;
                        case Access::read_write: usage = MTLResourceUsageRead | MTLResourceUsageWrite; break;
                        default: throw std::invalid_argument("invalid buffer access");
                    }
                    node.buffers.push_back(b.buffer.impl_);
                    node.offsets.push_back(b.offset);
                    p->resources[b.buffer.impl_->value] |= usage;
                }
                max_bindings = std::max(max_bindings, d.bindings.size());
            }
            const bool copy = node.copy_bytes != 0;
            const auto kind = copy ? detail::BatchKind::copy : detail::BatchKind::compute;
            if (p->batches.empty() || p->batches.back().kind != kind)
                p->batches.push_back({p->nodes.size(), 0, p->dispatch_count, kind});
            ++p->batches.back().count;
            if (!copy) ++p->dispatch_count;
            p->nodes.push_back(std::move(node));
        }
        // Task ids are plan-wide; a program contains a subset of them, each signalled then waited.
        for (int state : task_state)
            require(state != 1, "external task needs one signal followed by one wait");
        if (!p->signal_order.empty()) {
            p->event = [impl_->device newSharedEvent];
            if (!p->event) throw failure("create shared event", nil);
            p->event.label = p->label;
        }
        if (p->dispatch_count) {
            MTLIndirectCommandBufferDescriptor* descriptor = [MTLIndirectCommandBufferDescriptor new];
            descriptor.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
            descriptor.inheritPipelineState = NO;
            descriptor.inheritBuffers = NO;
            descriptor.maxKernelBufferBindCount = max_bindings;
            p->icb = [impl_->device newIndirectCommandBufferWithDescriptor:descriptor
                maxCommandCount:p->dispatch_count options:0];
            if (!p->icb) throw failure("create indirect command buffer", nil);
            p->icb.label = p->label;
            std::size_t command_index = 0;
            for (std::size_t i = 0; i < p->nodes.size(); ++i) {
                const auto& n = p->nodes[i];
                if (n.copy_bytes || n.external) continue;
                id<MTLIndirectComputeCommand> command = [p->icb indirectComputeCommandAtIndex:command_index++];
                if (i && !p->nodes[i - 1].copy_bytes && !p->nodes[i - 1].external) [command setBarrier];
                [command setComputePipelineState:n.kernel->value];
                for (std::size_t j = 0; j < n.buffers.size(); ++j)
                    [command setKernelBuffer:n.buffers[j]->value offset:n.offsets[j] atIndex:j];
                [command concurrentDispatchThreadgroups:n.groups threadsPerThreadgroup:n.threads];
            }
        }
        return Pipeline(std::move(p));
    }
}
std::size_t Pipeline::dispatch_count() const { return impl_->dispatch_count; }
std::size_t Pipeline::external_tasks() const { return impl_->signal_order.size(); }
Timing Pipeline::run(Execution mode) const { return run(mode, ExternalTask{}); }
Timing Pipeline::run(Execution mode, const ExternalTask& external) const {
    require(mode == Execution::direct || mode == Execution::icb, "invalid execution mode");
    require(impl_->signal_order.empty() || external, "program with external operations needs an executor");
    std::lock_guard<std::mutex> lock(impl_->context->mutex);
    const std::uint64_t tasks = impl_->signal_order.size();
    const std::uint64_t base = impl_->event_base;
    impl_->event_base += tasks ? 2 * tasks + 2 : 0;
    @autoreleasepool {
        const auto start = Clock::now();
        // An ExternalSignal ends the current command buffer, which is committed at once: its
        // completion (a reliable GPU->host notification) tells the host the task may start,
        // while the GPU continues with the next, already committed command buffer. The host
        // releases the matching ExternalWait through the shared event.
        std::vector<id<MTLCommandBuffer>> commands;
        std::vector<id<MTLCommandBuffer>> signalled(tasks ? *std::max_element(impl_->signal_order.begin(),
                                                                              impl_->signal_order.end()) + 1 : 0);
        double submitted = 0;
        auto begin = [&]() {
            id<MTLCommandBuffer> next = [impl_->context->queue commandBuffer];
            if (!next) throw failure("create command buffer", nil);
            next.label = [impl_->label stringByAppendingString:
                          mode == Execution::icb ? @" / ICB" : @" / direct"];
            commands.push_back(next);
            return next;
        };
        auto commit = [&](id<MTLCommandBuffer> buffer) {
            if (!submitted) submitted = host_seconds();
            [buffer commit];
        };
        id<MTLCommandBuffer> command = begin();
        // Batches are fixed at construction. The production compute-only plan
        // still executes one ICB range in one encoder; snapshots add blit passes.
        for (const auto& batch : impl_->batches) {
            if (batch.kind == detail::BatchKind::signal) {
                signalled[impl_->nodes[batch.begin].task] = command;
                commit(command);
                command = begin();
                continue;
            }
            if (batch.kind == detail::BatchKind::wait) {
                [command encodeWaitForEvent:impl_->event value:base + 2 * impl_->nodes[batch.begin].task + 2];
                continue;
            }
            if (batch.kind == detail::BatchKind::copy) {
                id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
                if (!blit) throw failure("create blit encoder", nil);
                blit.label = @"FlashRT buffer copies";
                for (std::size_t i = batch.begin; i < batch.begin + batch.count; ++i) {
                    const auto& n = impl_->nodes[i];
                    [blit copyFromBuffer:n.buffers[0]->value sourceOffset:n.offsets[0]
                        toBuffer:n.buffers[1]->value destinationOffset:n.offsets[1] size:n.copy_bytes];
                }
                [blit endEncoding];
                continue;
            }
            id<MTLComputeCommandEncoder> encoder =
                [command computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
            if (!encoder) throw failure("create compute encoder", nil);
            encoder.label = impl_->label;
            if (mode == Execution::icb) {
                for (const auto& resource : impl_->resources)
                    [encoder useResource:resource.first usage:resource.second];
                [encoder useResource:impl_->icb usage:MTLResourceUsageRead];
                [encoder executeCommandsInBuffer:impl_->icb withRange:NSMakeRange(batch.icb_begin, batch.count)];
            } else {
                for (std::size_t i = batch.begin; i < batch.begin + batch.count; ++i) {
                    const auto& n = impl_->nodes[i];
                    if (i != batch.begin) [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    if (n.label.length) [encoder pushDebugGroup:n.label];
                    [encoder setComputePipelineState:n.kernel->value];
                    for (std::size_t j = 0; j < n.buffers.size(); ++j)
                        [encoder setBuffer:n.buffers[j]->value offset:n.offsets[j] atIndex:j];
                    [encoder dispatchThreadgroups:n.groups threadsPerThreadgroup:n.threads];
                    if (n.label.length) [encoder popDebugGroup];
                }
            }
            [encoder endEncoding];
        }
        const auto encoded = Clock::now();
        commit(command);
        const auto committed = Clock::now();
        bool failed = false;
        std::string message;
        for (std::uint32_t task : impl_->signal_order) {
            id<MTLCommandBuffer> ready = signalled[task];
            [ready waitUntilCompleted];
            if (ready.status != MTLCommandBufferStatusCompleted) {
                failed = true; message = "GPU work before external task failed"; break;
            }
            std::string error;
            if (!external(task, error)) { failed = true; message = error.empty() ? "external task failed" : error; }
            impl_->event.signaledValue = base + 2 * task + 2;
            if (failed) break;
        }
        // Release every remaining wait so the GPU never blocks on a skipped task.
        if (failed && tasks) impl_->event.signaledValue = base + 2 * tasks + 2;
        id<MTLCommandBuffer> first = commands.front(), last = commands.back();
        [last waitUntilCompleted];
        const double returned = host_seconds();
        const auto end = Clock::now();
        for (id<MTLCommandBuffer> buffer : commands) {
            [buffer waitUntilCompleted];
            if (buffer.status != MTLCommandBufferStatusCompleted) throw failure("execute pipeline", buffer.error);
        }
        if (failed) throw std::runtime_error("external task: " + message);
        return {elapsed(start, encoded), (last.GPUEndTime-first.GPUStartTime)*1e6,
                elapsed(start, end), elapsed(encoded, committed), elapsed(committed, end),
                (first.GPUStartTime-submitted)*1e6, (returned-last.GPUEndTime)*1e6,
                (last.kernelEndTime-first.kernelStartTime)*1e6,
                submitted, first.GPUStartTime, last.GPUEndTime, returned};
    }
}
} // namespace flashrt::apple
