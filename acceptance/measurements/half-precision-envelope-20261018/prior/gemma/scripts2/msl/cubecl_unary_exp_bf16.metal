
#include <metal_stdlib>
using namespace metal;

struct alignas(8) bfloat_4 {
    bfloat i_0;
    bfloat i_1;
    bfloat i_2;
    bfloat i_3;
};

struct info_st {
    uint scalars_uint[2];uint static_meta[2];
};

[[kernel]]
void unary_float_f_bf16_n_4(
    device bfloat_4 * buffer_0 [[buffer(0)]],
    constant info_st& info [[buffer(1)]],
    uint3 thread_pos_in_grid [[thread_position_in_grid]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]]
) {
uint thread_index_in_grid = (
                thread_pos_in_grid.z * threadgroups_per_grid.x * threads_per_threadgroup.x * threadgroups_per_grid.y * threads_per_threadgroup.y)
                + (thread_pos_in_grid.y * threadgroups_per_grid.x * threads_per_threadgroup.x)
                + thread_pos_in_grid.x;
const bool l_0 = thread_index_in_grid < info.scalars_uint[1];
const uint l_1 = info.static_meta[uint(0)];
const bool l_2 = thread_index_in_grid < l_1;
const bool l_3 = l_0 && l_2;
const bool l_4 = !l_3;
if (l_4) {
return;}
const bfloat_4 l_5 = buffer_0[thread_index_in_grid];
const bfloat_4 l_6 = bfloat_4{
exp(l_5.i_0),exp(l_5.i_1),exp(l_5.i_2),exp(l_5.i_3),};
buffer_0[thread_index_in_grid] = reinterpret_cast<thread bfloat_4 const&>(l_6);

}
