#include <stdio.h>
#include <riscv_vector.h>

int main() {
    size_t vl = __riscv_vsetvl_e32m1(4);
    printf("Vector length: %zu\n", vl);
    
    float data[4] = {1.0, 2.0, 3.0, 4.0};
    vfloat32m1_t vec = __riscv_vle32_v_f32m1(data, vl);
    
    printf("RVV 1.0 test passed\n");
    return 0;
}
