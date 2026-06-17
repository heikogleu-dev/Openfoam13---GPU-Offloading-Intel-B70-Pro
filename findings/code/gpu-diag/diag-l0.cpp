// Pure Level Zero diagnostic - tests L0 layer without SYCL or Ginkgo
// Exit codes:
//   0 = ALL OK
//   1 = zeInit failed
//   2 = zeDriverGet failed
//   3 = zeDeviceGet failed
//   4 = zeContextCreate failed
//   5 = zeMemAllocDevice failed
//   6 = zeMemFree failed

#include <level_zero/ze_api.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(call, errcode, msg) do { \
    ze_result_t _r = (call); \
    if (_r != ZE_RESULT_SUCCESS) { \
        fprintf(stderr, "[FAIL] %s: 0x%x\n", msg, _r); \
        return errcode; \
    } else { \
        fprintf(stderr, "[ OK ] %s\n", msg); \
    } \
} while (0)

int main() {
    fprintf(stderr, "=== L0 Diagnostic ===\n");

    CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY), 1, "zeInit");

    uint32_t driver_count = 0;
    CHECK(zeDriverGet(&driver_count, nullptr), 2, "zeDriverGet (count)");
    fprintf(stderr, "[INFO] %u drivers found\n", driver_count);
    if (driver_count == 0) return 2;

    std::vector<ze_driver_handle_t> drivers(driver_count);
    CHECK(zeDriverGet(&driver_count, drivers.data()), 2, "zeDriverGet (handles)");

    ze_driver_handle_t driver = drivers[0];

    uint32_t device_count = 0;
    CHECK(zeDeviceGet(driver, &device_count, nullptr), 3, "zeDeviceGet (count)");
    fprintf(stderr, "[INFO] %u devices found\n", device_count);
    if (device_count == 0) return 3;

    std::vector<ze_device_handle_t> devices(device_count);
    CHECK(zeDeviceGet(driver, &device_count, devices.data()), 3, "zeDeviceGet (handles)");

    for (uint32_t i = 0; i < device_count; ++i) {
        ze_device_properties_t props = {ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
        if (zeDeviceGetProperties(devices[i], &props) == ZE_RESULT_SUCCESS) {
            fprintf(stderr, "[INFO] Device %u: %s (type=%d)\n", i, props.name, props.type);
        }
    }

    // Use device 0 (B70)
    ze_device_handle_t device = devices[0];

    ze_context_desc_t ctx_desc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC};
    ze_context_handle_t ctx = nullptr;
    CHECK(zeContextCreate(driver, &ctx_desc, &ctx), 4, "zeContextCreate");

    // Try a SMALL allocation
    void* ptr = nullptr;
    ze_device_mem_alloc_desc_t alloc_desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
    CHECK(zeMemAllocDevice(ctx, &alloc_desc, 1024, 0, device, &ptr), 5, "zeMemAllocDevice (1 KiB)");
    fprintf(stderr, "[INFO] Allocated 1 KiB at %p\n", ptr);

    CHECK(zeMemFree(ctx, ptr), 6, "zeMemFree");

    // Try a LARGER allocation (1 MiB)
    CHECK(zeMemAllocDevice(ctx, &alloc_desc, 1024*1024, 0, device, &ptr), 5, "zeMemAllocDevice (1 MiB)");
    CHECK(zeMemFree(ctx, ptr), 6, "zeMemFree");

    // Even larger (100 MiB)
    CHECK(zeMemAllocDevice(ctx, &alloc_desc, 100UL*1024*1024, 0, device, &ptr), 5, "zeMemAllocDevice (100 MiB)");
    CHECK(zeMemFree(ctx, ptr), 6, "zeMemFree");

    // Cleanup
    zeContextDestroy(ctx);

    fprintf(stderr, "[PASS] All L0 tests passed\n");
    return 0;
}
