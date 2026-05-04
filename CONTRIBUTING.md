# Contributing

Thanks for considering a contribution! This repo documents real-world
testing of OpenFOAM GPU acceleration on Intel Arc Pro B70 Pro
(Battlemage). Contributions are very welcome — particularly:

## High-Value Contributions

- **Reproducing a finding on different hardware** (Arc A-series, B580,
  B570, or other Battlemage variants) — open an issue with your
  numbers so we can compare
- **Reporting a different result on the same hardware** — environment
  differences (kernel, driver versions, BIOS) are interesting data
- **Bug reproductions** with stack traces / dmesg output for any
  finding numbered 01–12
- **Performance updates** when a new Intel Compute Runtime, oneAPI,
  or OGL/Ginkgo version changes the picture
- **Workarounds for documented bugs** — especially for SYCL preconditioner
  issues (BJ maxBlockSize, ICT, Multigrid)
- **Successful Ginkgo 2.0 migration** of OGL — see
  [findings/10](findings/10_ginkgo2_api_breaks.md)

## How to Contribute

1. **Open an issue first** for non-trivial changes — it's faster to
   align on direction than to rework a large PR
2. Fork, branch, edit, push, open a PR
3. Reference the relevant `findings/##_*.md` file(s) in your PR
4. Include reproduction steps for any new benchmark / measurement

## Documentation Style

- Real numbers, not "feels faster". Steady-state mean of 3 timesteps.
- Include hardware + software versions (kernel, driver, oneAPI, OGL/Ginkgo)
- Stack traces + dmesg context for crashes
- Honest verdicts — negative results are valuable

## License

By contributing, you agree your contributions are licensed under the
[GPL v3](LICENSE), the same license as this repository.
