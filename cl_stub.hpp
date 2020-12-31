template<uint x>
size_t __get_global_id(){
  if constexpr(x == 0)
    return __nvvm_read_ptx_sreg_tid_x() + __nvvm_read_ptx_sreg_ctaid_x() * __nvvm_read_ptx_sreg_ntid_x();
  else if constexpr(x == 1)
    return __nvvm_read_ptx_sreg_tid_y() + __nvvm_read_ptx_sreg_ctaid_y() * __nvvm_read_ptx_sreg_ntid_y();
  else if constexpr(x == 2)
    return __nvvm_read_ptx_sreg_tid_z() + __nvvm_read_ptx_sreg_ctaid_z() * __nvvm_read_ptx_sreg_ntid_z();
  else
    static_assert(x <= 2);
}

#define get_global_id(x) __get_global_id<x>()

float __min(float x, float y){
  return x < y ? x : y;
}

#define min __min

float __max(float x, float y){
  return x > y ? x : y;
}

#define max __max
