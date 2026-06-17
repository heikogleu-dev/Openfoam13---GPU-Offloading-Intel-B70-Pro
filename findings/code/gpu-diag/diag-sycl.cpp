// SYCL diagnostic - tests SYCL queue + memory + kernel without Ginkgo
// Exit codes:
//   0 = ALL OK
//   1 = sycl::device GPU not found
//   2 = sycl::queue create failed
//   3 = sycl::malloc_device failed
//   4 = sycl::memset failed
//   5 = sycl kernel failed

#include <sycl/sycl.hpp>
#include <cstdio>
#include <stdexcept>

int main() {
    fprintf(stderr, "=== SYCL Diagnostic ===\n");

    sycl::device gpu;
    try {
        gpu = sycl::device(sycl::gpu_selector_v);
        fprintf(stderr, "[ OK ] sycl::device GPU: %s\n",
            gpu.get_info<sycl::info::device::name>().c_str());
    } catch (sycl::exception& e) {
        fprintf(stderr, "[FAIL] sycl::device GPU: %s\n", e.what());
        return 1;
    }

    sycl::queue q;
    try {
        q = sycl::queue(gpu, sycl::property::queue::in_order());
        fprintf(stderr, "[ OK ] sycl::queue created on %s\n",
            q.get_device().get_info<sycl::info::device::name>().c_str());
    } catch (sycl::exception& e) {
        fprintf(stderr, "[FAIL] sycl::queue: %s\n", e.what());
        return 2;
    }

    // Small allocation
    int* p_small = nullptr;
    try {
        p_small = sycl::malloc_device<int>(256, q);
        fprintf(stderr, "[ OK ] sycl::malloc_device (1 KiB) at %p\n", (void*)p_small);
    } catch (sycl::exception& e) {
        fprintf(stderr, "[FAIL] sycl::malloc_device 1 KiB: %s\n", e.what());
        return 3;
    }

    // memset
    try {
        q.memset(p_small, 0, 256*sizeof(int)).wait();
        fprintf(stderr, "[ OK ] sycl::memset\n");
    } catch (sycl::exception& e) {
        fprintf(stderr, "[FAIL] sycl::memset: %s\n", e.what());
        sycl::free(p_small, q);
        return 4;
    }

    // Tiny kernel
    try {
        q.parallel_for(sycl::range<1>(256), [=](sycl::id<1> i) {
            p_small[i] = (int)i.get(0);
        }).wait();
        fprintf(stderr, "[ OK ] sycl::parallel_for (256 work-items)\n");
    } catch (sycl::exception& e) {
        fprintf(stderr, "[FAIL] sycl::parallel_for: %s\n", e.what());
        sycl::free(p_small, q);
        return 5;
    }

    sycl::free(p_small, q);

    // Bigger allocation
    int* p_big = nullptr;
    try {
        size_t big = 100UL * 1024 * 1024 / sizeof(int);  // 100 MiB
        p_big = sycl::malloc_device<int>(big, q);
        fprintf(stderr, "[ OK ] sycl::malloc_device (100 MiB) at %p\n", (void*)p_big);
        sycl::free(p_big, q);
    } catch (sycl::exception& e) {
        fprintf(stderr, "[FAIL] sycl::malloc_device 100 MiB: %s\n", e.what());
        return 3;
    }

    fprintf(stderr, "[PASS] All SYCL tests passed\n");
    return 0;
}
