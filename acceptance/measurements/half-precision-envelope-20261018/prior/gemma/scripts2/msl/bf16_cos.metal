#include <metal_stdlib>
using namespace metal;
kernel void k(device bfloat* x [[buffer(0)]], uint i [[thread_position_in_grid]]) { x[i] = cos(x[i]); }
