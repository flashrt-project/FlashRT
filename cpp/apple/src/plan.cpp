#include <flashrt/apple/plan.h>
#include <flashrt/apple/pipeline.h>
#include "kernel_registry.h"
#include <map>
#include <mutex>
#include <stdexcept>
using namespace flashrt::apple;
namespace {
thread_local std::string error;
struct Plan {
    Context context;
    std::mutex mutex;
    std::vector<Buffer> buffers;
    std::map<std::string,Kernel> kernels;
    std::vector<Operation> operations;
    std::unique_ptr<Pipeline> pipeline;
    std::vector<Pipeline> profiles;
};
void require(bool c,const char* m) { if(!c) throw std::invalid_argument(m); }
Plan& checked(void* p) { require(p,"null plan");return *static_cast<Plan*>(p); }
template<class F> int guard(F f) {
    try { f();error.clear();return 0; }
    catch(const std::exception& e) {error=e.what();return -1;}
    catch(...) {error="unknown plan failure";return -1;}
}
}
extern "C" {
const char* frt_apple_plan_error() {return error.c_str();}
void* frt_apple_plan_create() {void* p=nullptr;guard([&]{p=new Plan;});return p;}
void frt_apple_plan_destroy(void* p) {delete static_cast<Plan*>(p);}
int64_t frt_apple_plan_buffer(void* p,size_t n) {
    int64_t id=-1;guard([&]{auto& x=checked(p);std::lock_guard<std::mutex> lock(x.mutex);
        require(!x.pipeline,"plan already sealed");id=x.buffers.size();x.buffers.push_back(x.context.buffer(n));});return id;
}
int frt_apple_plan_write(void* p,uint32_t id,size_t off,const void* src,size_t n) {
    return guard([&]{auto& x=checked(p);std::lock_guard<std::mutex> lock(x.mutex);x.buffers.at(id).write(src,n,off);});
}
int frt_apple_plan_read(void* p,uint32_t id,size_t off,void* dst,size_t n) {
    return guard([&]{auto& x=checked(p);std::lock_guard<std::mutex> lock(x.mutex);x.buffers.at(id).read(dst,n,off);});
}
void* frt_apple_plan_data(void* p,uint32_t id,size_t off,size_t n) {
    void* result=nullptr;
    guard([&]{auto& x=checked(p);std::lock_guard<std::mutex> lock(x.mutex);
        auto& b=x.buffers.at(id);require(off<=b.size() && n<=b.size()-off,"mapping out of bounds");
        result=static_cast<char*>(b.host_data())+off;});
    return result;
}
int frt_apple_plan_dispatch(void* p,const char* name,const uint32_t* ids,const size_t* offsets,
    const uint32_t* access,size_t n,const size_t* groups,size_t threads,const char* label) {
    return guard([&]{auto& x=checked(p);std::lock_guard<std::mutex> lock(x.mutex);
        require(!x.pipeline && name && ids && offsets && access && groups && label,"invalid plan dispatch");
        require(n>0 && n<=31,"invalid binding count");
        std::string key(name);
        if(!x.kernels.count(key)) {
            const auto& source=shaders::find_kernel_source(key);
            x.kernels.emplace(key,x.context.kernel(source.build_source(),source.entry,true));
        }
        std::vector<Binding> bindings;
        for(size_t i=0;i<n;++i) {
            require(access[i]<=2,"invalid binding access");
            bindings.push_back({x.buffers.at(ids[i]),offsets[i],static_cast<Access>(access[i])});
        }
        x.operations.push_back(Dispatch{x.kernels.at(key),bindings,{groups[0],groups[1],groups[2]},
            {threads},label});
    });
}
int frt_apple_plan_copy(void* p,uint32_t source,size_t source_offset,
    uint32_t destination,size_t destination_offset,size_t bytes) {
    return guard([&]{auto& x=checked(p);std::lock_guard<std::mutex> lock(x.mutex);
        require(!x.pipeline,"plan already sealed");
        x.operations.push_back(BufferCopy{{x.buffers.at(source),source_offset,Access::read},
            {x.buffers.at(destination),destination_offset,Access::write},bytes,"buffer copy"});
    });
}
int frt_apple_plan_finish(void* p) {return guard([&]{auto& x=checked(p);std::lock_guard<std::mutex> lock(x.mutex);
    require(!x.pipeline,"plan already sealed");x.pipeline=std::make_unique<Pipeline>(x.context.program(x.operations,"FlashRT fixed Metal plan"));});}
int frt_apple_plan_residency(void* p,uint32_t enabled) {
    return guard([&]{auto& x=checked(p);std::lock_guard<std::mutex> lock(x.mutex);
        require(x.pipeline && enabled<=1,"invalid residency request");x.pipeline->set_resident(enabled);});
}
int frt_apple_plan_run_timed(void* p,uint32_t mode,frt_apple_plan_timing* timing) {
    return guard([&]{auto& x=checked(p);std::lock_guard<std::mutex> lock(x.mutex);
        require(x.pipeline && mode<=1 && timing,"invalid timed run");
        auto t=x.pipeline->run(mode ? Execution::icb:Execution::direct);
        *timing={t.encode_us,t.gpu_us,t.wall_us,t.commit_us,t.wait_us,
            t.commit_to_gpu_us,t.gpu_to_return_us,t.kernel_us,t.submit_host_s,
            t.gpu_start_s,t.gpu_end_s,t.return_host_s};});
}
int frt_apple_plan_profile(void* p,frt_apple_plan_timing* timing,size_t count) {
    return guard([&]{auto& x=checked(p);std::lock_guard<std::mutex> lock(x.mutex);
        require(x.pipeline && timing && count==x.operations.size(),"invalid profile output");
        if(x.profiles.empty()) {
            std::vector<Pipeline> profiles;
            for(const auto& op:x.operations) profiles.push_back(x.context.program({op},
                std::visit([](const auto& value){return value.label;},op)));
            x.profiles=std::move(profiles);
        }
        for(size_t i=0;i<count;++i) {
            auto t=x.profiles[i].run(Execution::direct);
            timing[i]={t.encode_us,t.gpu_us,t.wall_us,t.commit_us,t.wait_us,
                t.commit_to_gpu_us,t.gpu_to_return_us,t.kernel_us,t.submit_host_s,
                t.gpu_start_s,t.gpu_end_s,t.return_host_s};
        }
    });
}
}
