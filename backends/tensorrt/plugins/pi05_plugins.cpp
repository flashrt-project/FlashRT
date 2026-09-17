// Plugin library entry points: every pi0.5 creator in one shared object.
#include <NvInferRuntime.h>

nvinfer1::IPluginCreatorInterface* flashrt_trt_pi05_encoder_layer_creator();
nvinfer1::IPluginCreatorInterface* flashrt_trt_pi05_decoder_step_creator();
nvinfer1::IPluginCreatorInterface* flashrt_trt_pi05_decoder_creator();
nvinfer1::IPluginCreatorInterface* flashrt_trt_pi05_encoder_creator();
nvinfer1::IPluginCreatorInterface* flashrt_trt_pi05_siglip_layer_creator();
nvinfer1::IPluginCreatorInterface* flashrt_trt_pi05_siglip_creator();

extern "C" void setLoggerFinder(nvinfer1::ILoggerFinder*) {}

extern "C" nvinfer1::IPluginCreatorInterface* const* getCreators(int32_t& nbCreators) {
    static nvinfer1::IPluginCreatorInterface* const creators[] = {
        flashrt_trt_pi05_encoder_layer_creator(),
        flashrt_trt_pi05_decoder_step_creator(),
        flashrt_trt_pi05_decoder_creator(),
        flashrt_trt_pi05_encoder_creator(),
        flashrt_trt_pi05_siglip_layer_creator(),
        flashrt_trt_pi05_siglip_creator(),
    };
    nbCreators = 6;
    return creators;
}
