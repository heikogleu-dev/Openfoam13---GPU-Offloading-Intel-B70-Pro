# OGL ExecutorHandler.hpp — MPI_Barrier Patch (attempted, insufficient)

This is the patch tried against `hpsim/OGL` PR #168 branch +
ExecutorHandler.hpp to address the SYCL multi-rank init race on
Intel BMG-G31 / CR 26.18. **Result: insufficient for the OGL +
OpenFOAM stack** — see [finding 28](../../28_gpu_layer_diagnostic_tool_findings.md).

The barrier suppresses the race for minimal SYCL/Ginkgo programs
(verified with the diag-mpi-ginkgo tool) but does not reach the
earlier static-init phase where libOGL.so + libginkgo* SOs trigger
the OpenCL adapter load. Kept here for upstream reference.

## Diff vs OGL PR #168 baseline

```diff
diff --git a/include/OGL/DevicePersistent/ExecutorHandler.hpp b/include/OGL/DevicePersistent/ExecutorHandler.hpp

@@ -9,6 +9,7 @@
 #include "fvCFD.H"

 #include <ginkgo/ginkgo.hpp>
+#include <mpi.h>

 namespace Foam {

@@ -166,9 +167,28 @@
                        "with SYCL backend enabled."
                     << abort(FatalError);
             }
+            // Workaround attempt for SYCL multi-rank init race on Intel
+            // BMG-G31 + Compute Runtime 26.18 (see findings/28).
+            // get_num_devices() triggers SYCL platform enumeration which
+            // loads the OpenCL adapter and hits a pthread_once race in
+            // clGetPlatformIDs when invoked concurrently. Insert MPI
+            // barriers around the two SYCL-touching calls.
+            // INSUFFICIENT for the full OpenFOAM+OGL stack — race is
+            // reached earlier (likely SO static-init phase).
+            if (Pstream::parRun()) {
+                MPI_Barrier(MPI_COMM_WORLD);
+            }
             label id = device_id_handler_.compute_device_id(
                 gko::DpcppExecutor::get_num_devices("gpu"));
             LOG_0(verbose_, msg(executor_name_, id))
+            if (Pstream::parRun()) {
+                MPI_Barrier(MPI_COMM_WORLD);
+            }
             return gko::share(gko::DpcppExecutor::create(id, host_exec));
         }
```
