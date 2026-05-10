# NOTICE — Third-Party Software Attribution

This repository (own work — READMEs, findings, build documentation, configs)
is licensed under **GPL-3.0-or-later** (see [LICENSE](LICENSE)).

It **does not redistribute** any third-party source code or binaries. It
documents how to fetch and build upstream software locally, and provides
diagnostic logs and configuration templates that describe behavior of
*your* local copies of the listed components.

---

## Build-time Dependencies (User Fetches & Installs Locally)

| Project | License | Source URL |
|---|---|---|
| **OpenFOAM Foundation v13** | GPL-3.0-or-later | https://github.com/OpenFOAM/OpenFOAM-13 |
| **Ginkgo 1.10 / 1.11** | BSD-3-Clause | https://github.com/ginkgo-project/ginkgo |
| **OGL** (Ginkgo OpenFOAM Layer) | GPL-3.0-or-later | https://github.com/hpsim/OGL |
| **Intel oneAPI Base Toolkit 2025.3 / 2026.0** | Intel Simplified Software License (binary EULA) | https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html |
| **Intel Compute Runtime / Level-Zero** | MIT (NEO + libze) | https://github.com/intel/compute-runtime / https://github.com/oneapi-src/level-zero |
| **Ubuntu 26.04 LTS** | various (Debian-derived) | https://ubuntu.com/ |

---

## License Compatibility

GPL-3.0-or-later is the umbrella license of this repository. Inclusion of:

- **OpenFOAM (GPL-3):** identical license, fully compatible.
- **Ginkgo (BSD-3):** permissive, compatible downstream into GPL-3 work.
- **OGL (GPL-3):** identical license, fully compatible.
- **Intel oneAPI:** only compiled artifacts and configure-log output are
  included; no Intel proprietary source code is redistributed.

---

## Build Logs and Configure Output

Build logs in `logs/` directories contain compiler output, library paths,
and runtime diagnostic messages. These are mechanical build artifacts and
do not embed redistributable source code.

---

## Test/Diagnostic Code

Standalone diagnostic test code authored for this repository is licensed
under GPL-3.0-or-later as part of the repo. The reproducer source for the
2026-05-10 cross-stack diagnostic lives outside the repo (under
`/home/heiko/diag/` on the workstation), not committed.

---

## Cross-Reference

The petsc4Foam sister repo carries an analogous attribution scheme:
[Openfoam-v2512-Petsc-Kokkos-Sycl-Intel-B70/NOTICE.md](https://github.com/heikogleu-dev/Openfoam-v2512-Petsc-Kokkos-Sycl-Intel-B70/blob/main/NOTICE.md)
