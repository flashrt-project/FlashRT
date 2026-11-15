#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include "coreml.h"
#include <cstring>
#include <stdexcept>

namespace flashrt::apple {
struct CoreMLTask::Impl {
    MLModel* model = nil;
    id<MLFeatureProvider> inputs = nil;
    MLPredictionOptions* options = nil;
    MLMultiArray* output_view = nil;
    NSString* output = nil;
    void* output_data = nullptr;
    std::size_t bytes = 0;
    mutable bool backed = false;
};

namespace {
std::runtime_error coreml_failure(const char* what, NSError* error) {
    return std::runtime_error(std::string("Core ML ") + what + ": " +
                              (error ? error.localizedDescription.UTF8String : "returned nil"));
}
MLMultiArray* view(void* data, NSArray<NSNumber*>* shape, NSArray<NSNumber*>* strides) {
    NSError* error = nil;
    MLMultiArray* array = [[MLMultiArray alloc] initWithDataPointer:data shape:shape
        dataType:MLMultiArrayDataTypeFloat16 strides:strides deallocator:nil error:&error];
    if (!array) throw coreml_failure("multi-array view", error);
    return array;
}
}

CoreMLTask::CoreMLTask(const std::string& model_path, const std::string& input, void* input_data,
                       const std::string& output, void* output_data, const std::vector<std::int64_t>& shape,
                       int compute_units) : impl_(std::make_unique<Impl>()) {
    if (!input_data || !output_data || shape.empty()) throw std::invalid_argument("invalid Core ML task views");
    @autoreleasepool {
        NSError* error = nil;
        NSString* path = [NSString stringWithUTF8String:model_path.c_str()];
        NSURL* url = [NSURL fileURLWithPath:path];
        if (![path hasSuffix:@".mlmodelc"]) {
            url = [MLModel compileModelAtURL:url error:&error];
            if (!url) throw coreml_failure("compile", error);
        }
        MLModelConfiguration* config = [MLModelConfiguration new];
        const MLComputeUnits units[] = {MLComputeUnitsCPUAndNeuralEngine, MLComputeUnitsAll,
                                        MLComputeUnitsCPUAndGPU, MLComputeUnitsCPUOnly};
        if (compute_units < 0 || compute_units > 3) throw std::invalid_argument("invalid Core ML compute units");
        config.computeUnits = units[compute_units];
        impl_->model = [MLModel modelWithContentsOfURL:url configuration:config error:&error];
        if (!impl_->model) throw coreml_failure("load", error);
        NSMutableArray<NSNumber*>* dims = [NSMutableArray array];
        NSMutableArray<NSNumber*>* strides = [NSMutableArray array];
        std::int64_t count = 1;
        for (auto it = shape.rbegin(); it != shape.rend(); ++it) {
            if (*it <= 0) throw std::invalid_argument("invalid Core ML task shape");
            [strides insertObject:@(count) atIndex:0];
            count *= *it;
        }
        for (auto d : shape) [dims addObject:@(d)];
        impl_->bytes = std::size_t(count) * 2;
        NSString* in = [NSString stringWithUTF8String:input.c_str()];
        impl_->output = [NSString stringWithUTF8String:output.c_str()];
        impl_->inputs = [[MLDictionaryFeatureProvider alloc]
            initWithDictionary:@{in: [MLFeatureValue featureValueWithMultiArray:view(input_data, dims, strides)]}
                         error:&error];
        if (!impl_->inputs) throw coreml_failure("inputs", error);
        impl_->output_data = output_data;
        impl_->output_view = view(output_data, dims, strides);
        impl_->options = [MLPredictionOptions new];
        impl_->options.outputBackings = @{impl_->output: impl_->output_view};
    }
}

CoreMLTask::~CoreMLTask() = default;

bool CoreMLTask::output_backed() const { return impl_->backed; }

bool CoreMLTask::run(std::string& message) const {
    @autoreleasepool {
        NSError* error = nil;
        id<MLFeatureProvider> result = [impl_->model predictionFromFeatures:impl_->inputs
                                                                    options:impl_->options error:&error];
        if (!result) {
            message = std::string("Core ML prediction: ") + (error ? error.localizedDescription.UTF8String : "failed");
            return false;
        }
        MLMultiArray* value = [result featureValueForName:impl_->output].multiArrayValue;
        if (!value || value.dataType != MLMultiArrayDataTypeFloat16 ||
            std::size_t(value.count) * 2 != impl_->bytes) {
            message = "Core ML prediction: unexpected output";
            return false;
        }
        impl_->backed = value == impl_->output_view;
        if (!impl_->backed) {
            // Framework-owned output, possibly with padded strides: gather into the dense view.
            NSArray<NSNumber*>* shape = value.shape;
            NSArray<NSNumber*>* strides = value.strides;
            const NSUInteger rank = shape.count;
            __block bool copied = false;
            [value getBytesWithHandler:^(const void* bytes, NSInteger size) {
                const auto* src = static_cast<const std::uint16_t*>(bytes);
                auto* dst = static_cast<std::uint16_t*>(impl_->output_data);
                std::vector<std::int64_t> dims(rank), step(rank), index(rank, 0);
                std::int64_t last = 0;
                for (NSUInteger i = 0; i < rank; ++i) {
                    dims[i] = shape[i].longLongValue; step[i] = strides[i].longLongValue;
                    last += (dims[i] - 1) * step[i];
                }
                if ((last + 1) * 2 > size) return;
                const std::int64_t inner = dims[rank - 1];
                const std::int64_t count = std::int64_t(impl_->bytes / 2);
                for (std::int64_t out = 0; out < count; out += inner) {
                    std::int64_t offset = 0;
                    for (NSUInteger i = 0; i + 1 < rank; ++i) offset += index[i] * step[i];
                    if (step[rank - 1] == 1) std::memcpy(dst + out, src + offset, std::size_t(inner) * 2);
                    else for (std::int64_t j = 0; j < inner; ++j) dst[out + j] = src[offset + j * step[rank - 1]];
                    for (NSInteger i = NSInteger(rank) - 2; i >= 0; --i) {
                        if (++index[i] < dims[i]) break;
                        index[i] = 0;
                    }
                }
                copied = true;
            }];
            if (!copied) { message = "Core ML prediction: output copy failed"; return false; }
        }
        return true;
    }
}
}  // namespace flashrt::apple
