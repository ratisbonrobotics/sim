#include "/home/markusheimerl/slang/prelude/slang-cuda-prelude.h"


#line 11 "hello_world.slang"
struct GlobalParams_0
{
    StructuredBuffer<float> buffer0_0;
    StructuredBuffer<float> buffer1_0;
    RWStructuredBuffer<float> result_0;
};


#line 11
extern "C" __constant__ GlobalParams_0 SLANG_globalParams;
#define globalParams_0 (&SLANG_globalParams)

#line 8
extern "C" __global__ void computeMain()
{
    uint index_0 = (blockIdx * blockDim + threadIdx).x;
    *(&(globalParams_0->result_0[index_0])) = globalParams_0->buffer0_0.Load(index_0) + globalParams_0->buffer1_0.Load(index_0);
    return;
}


#line 20140 "hlsl.meta.slang"
struct NullDifferential_0
{
    uint dummy_0;
};

