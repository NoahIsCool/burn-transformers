#include <metal_stdlib>
using namespace metal;
kernel void k(device half* x [[buffer(0)]], uint i [[thread_position_in_grid]]) { x[i] = exp(x[i]); }
