#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "fusion_jit.h"
#include <stdlib.h>
#include <string.h>

typedef struct MetalFusionJIT {
    void* device;
    void* queue;
    void* library;
    void* pipeline;
} MetalFusionJIT;

static void metal_fusion_status_set(MetalFusionStatus* status, int code, const char* message) {
    if (status == NULL) {
        return;
    }

    status->code = code;

    if (message == NULL) {
        status->message[0] = '\0';
        return;
    }

    strncpy(status->message, message, METAL_FUSION_STATUS_MESSAGE_BYTES - 1);
    status->message[METAL_FUSION_STATUS_MESSAGE_BYTES - 1] = '\0';
}

static void metal_fusion_status_set_ns_error(
    MetalFusionStatus* status,
    int code,
    NSString* operation,
    NSError* error
) {
    if (status == NULL) {
        return;
    }

    if (error == nil) {
        metal_fusion_status_set(status, code, [operation UTF8String]);
        return;
    }

    NSString* message = [NSString stringWithFormat:@"%@: %@", operation, error.localizedDescription];
    metal_fusion_status_set(status, code, message.UTF8String);
}

MetalFusionJITRef metal_fusion_jit_compile(
    const char* source,
    const char* kernelName,
    MetalFusionStatus* status
) {
    @autoreleasepool {
        metal_fusion_status_set(status, 0, NULL);

        if (source == NULL || kernelName == NULL) {
            metal_fusion_status_set(status, -1, "fusion jit: source and kernel name are required");
            return NULL;
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();

        if (device == nil) {
            metal_fusion_status_set(status, -2, "MTLCreateSystemDefaultDevice returned nil");
            return NULL;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];

        if (queue == nil) {
            metal_fusion_status_set(status, -3, "newCommandQueue returned nil");
            return NULL;
        }

        NSString* sourceText = [NSString stringWithUTF8String:source];
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:sourceText options:nil error:&error];

        if (library == nil) {
            metal_fusion_status_set_ns_error(status, -4, @"newLibraryWithSource", error);
            return NULL;
        }

        NSString* functionName = [NSString stringWithUTF8String:kernelName];
        id<MTLFunction> function = [library newFunctionWithName:functionName];

        if (function == nil) {
            metal_fusion_status_set(status, -5, "newFunctionWithName returned nil");
            return NULL;
        }

        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function error:&error];

        if (pipeline == nil) {
            metal_fusion_status_set_ns_error(status, -6, @"newComputePipelineStateWithFunction", error);
            return NULL;
        }

        MetalFusionJIT* jit = (MetalFusionJIT*)calloc(1, sizeof(MetalFusionJIT));

        if (jit == NULL) {
            metal_fusion_status_set(status, -7, "calloc MetalFusionJIT failed");
            return NULL;
        }

        jit->device = (__bridge_retained void*)device;
        jit->queue = (__bridge_retained void*)queue;
        jit->library = (__bridge_retained void*)library;
        jit->pipeline = (__bridge_retained void*)pipeline;

        return jit;
    }
}

void metal_fusion_jit_release(MetalFusionJITRef jitRef) {
    @autoreleasepool {
        MetalFusionJIT* jit = (MetalFusionJIT*)jitRef;

        if (jit == NULL) {
            return;
        }

        if (jit->pipeline != NULL) {
            CFRelease(jit->pipeline);
            jit->pipeline = NULL;
        }

        if (jit->library != NULL) {
            CFRelease(jit->library);
            jit->library = NULL;
        }

        if (jit->queue != NULL) {
            CFRelease(jit->queue);
            jit->queue = NULL;
        }

        if (jit->device != NULL) {
            CFRelease(jit->device);
            jit->device = NULL;
        }

        free(jit);
    }
}

static id<MTLBuffer> metal_fusion_buffer_from_host(
    id<MTLDevice> device,
    const float* values,
    uint32_t count
) {
    if (count == 0) {
        return nil;
    }

    id<MTLBuffer> buffer = [device newBufferWithBytes:values
                                               length:(NSUInteger)count * sizeof(float)
                                              options:MTLResourceStorageModeShared];

    return buffer;
}

int metal_fusion_jit_run_host(
    MetalFusionJITRef jitRef,
    const float** inputs,
    float* output,
    int inputCount,
    uint32_t count,
    MetalFusionStatus* status
) {
    @autoreleasepool {
        metal_fusion_status_set(status, 0, NULL);

        if (count == 0) {
            return 0;
        }

        MetalFusionJIT* jit = (MetalFusionJIT*)jitRef;

        if (jit == NULL || jit->device == NULL || jit->queue == NULL || jit->pipeline == NULL) {
            metal_fusion_status_set(status, -10, "invalid fusion jit handle");
            return -10;
        }

        if (inputs == NULL || output == NULL || inputCount < 0) {
            metal_fusion_status_set(status, -11, "invalid fusion jit buffers");
            return -11;
        }

        id<MTLDevice> device = (__bridge id<MTLDevice>)jit->device;
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)jit->queue;
        id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)jit->pipeline;

        NSMutableArray<id<MTLBuffer>>* inputBuffers =
            [NSMutableArray arrayWithCapacity:(NSUInteger)inputCount];

        for (int inputIndex = 0; inputIndex < inputCount; inputIndex++) {
            if (inputs[inputIndex] == NULL) {
                metal_fusion_status_set(status, -12, "nil fusion jit input buffer");
                return -12;
            }

            id<MTLBuffer> buffer = metal_fusion_buffer_from_host(device, inputs[inputIndex], count);

            if (buffer == nil) {
                metal_fusion_status_set(status, -13, "newBufferWithBytes for input failed");
                return -13;
            }

            [inputBuffers addObject:buffer];
        }

        id<MTLBuffer> outputBuffer = [device newBufferWithLength:(NSUInteger)count * sizeof(float)
                                                         options:MTLResourceStorageModeShared];

        if (outputBuffer == nil) {
            metal_fusion_status_set(status, -14, "newBufferWithLength for output failed");
            return -14;
        }

        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];

        if (commandBuffer == nil) {
            metal_fusion_status_set(status, -15, "commandBuffer returned nil");
            return -15;
        }

        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        if (encoder == nil) {
            metal_fusion_status_set(status, -16, "computeCommandEncoder returned nil");
            return -16;
        }

        [encoder setComputePipelineState:pipeline];

        for (NSUInteger bufferIndex = 0; bufferIndex < inputBuffers.count; bufferIndex++) {
            [encoder setBuffer:inputBuffers[bufferIndex] offset:0 atIndex:bufferIndex];
        }

        [encoder setBuffer:outputBuffer offset:0 atIndex:(NSUInteger)inputCount];
        [encoder setBytes:&count length:sizeof(count) atIndex:(NSUInteger)inputCount + 1];

        NSUInteger threadgroupWidth = pipeline.threadExecutionWidth;

        if (threadgroupWidth == 0) {
            threadgroupWidth = 256;
        }

        NSUInteger threadgroups = (count + (uint32_t)threadgroupWidth - 1) / (uint32_t)threadgroupWidth;
        [encoder dispatchThreadgroups:MTLSizeMake(threadgroups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threadgroupWidth, 1, 1)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
            metal_fusion_status_set(status, -17, "fusion jit command buffer did not complete");
            return -17;
        }

        memcpy(output, outputBuffer.contents, (size_t)count * sizeof(float));

        return 0;
    }
}
