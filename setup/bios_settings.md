# BIOS Settings — ASRock Z890I Nova WiFi

For optimal OpenFOAM GPU compute performance with Intel Arc Pro B70 Pro:

| Setting | Value | Notes |
|---|---|---|
| PCIe Bifurcation | x16 | GPU slot must be x16 |
| Above 4G Decoding | Enabled | Required for Resizable BAR |
| Resizable BAR | Enabled | Confirmed active (32 GB BAR) |
| iGPU | Enabled | Let iGPU drive display |
| Primary Display | CPU Graphics (iGPU) | B70 Pro = pure compute |
| XMP/DOCP | Enabled (DDR5-6800) | Critical for CPU memory bandwidth |
| Package C-State | C0/C1 | Prevents MPI wake-up latency |
| IO MWAIT Redirection | Enabled | Faster MPI barriers |
| DRAM Power Down Mode | Disabled | Reduces memory latency |
| DC6 Latency WA | Enabled | With iGPU active |
| PCIe ASPM | Disabled (or Performance) | Minimize wakeup latency |
| SR-IOV | Auto (active) | B70 Pro runs in PF mode by default |

## Display Setup (CRITICAL for Compute)

With iGPU enabled, connect monitor to **motherboard video output**.
This frees the B70 Pro from display overhead (~1 GB VRAM + Compositor cycles).

Configure via xorg.conf:

`/etc/X11/xorg.conf.d/20-igpu-display.conf`:

```
Section "Device"
    Identifier  "iGPU-Display"
    Driver      "modesetting"
    BusID       "PCI:0:2:0"
EndSection

Section "Screen"
    Identifier  "iGPU-Screen"
    Device      "iGPU-Display"
EndSection
```

Reboot for the xorg config to take effect.

## Verification

After reboot:
```bash
sycl-ls                         # Should still show both GPUs
xrandr --listproviders          # Display should be on iGPU
glxinfo | grep "OpenGL renderer" # Should mention Arrow Lake / iGPU
```

VRAM idle baseline (display on iGPU):
```bash
sudo cat /sys/kernel/debug/dri/0/tile0/vram_mm | grep usage
# Expected: <500 MB (vs ~2.3 GB if display is on B70 Pro)
```
