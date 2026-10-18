#include <metal_stdlib>
using namespace metal;
kernel void k(device bfloat* a [[buffer(0)]], device float* c [[buffer(1)]]) {
  simdgroup_matrix<bfloat, 8, 8> ma; simdgroup_matrix<bfloat, 8, 8> mb; simdgroup_matrix<float, 8, 8> mc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  simdgroup_load(ma, a, 8); simdgroup_load(mb, a, 8); simdgroup_multiply_accumulate(mc, ma, mb, mc); simdgroup_store(mc, c, 8);
}
