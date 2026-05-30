#ifndef MANIFESTO_CODEGEN_METAL_FUSION_JIT_H
#define MANIFESTO_CODEGEN_METAL_FUSION_JIT_H

#include <stddef.h>
#include <stdint.h>

#define METAL_FUSION_STATUS_MESSAGE_BYTES 1024

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MetalFusionStatus {
    int code;
    char message[METAL_FUSION_STATUS_MESSAGE_BYTES];
} MetalFusionStatus;

typedef void* MetalFusionJITRef;

MetalFusionJITRef metal_fusion_jit_compile(
    const char* source,
    const char* kernelName,
    MetalFusionStatus* status
);

void metal_fusion_jit_release(MetalFusionJITRef jitRef);

int metal_fusion_jit_run_host(
    MetalFusionJITRef jitRef,
    const float** inputs,
    float* output,
    int inputCount,
    uint32_t count,
    MetalFusionStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
