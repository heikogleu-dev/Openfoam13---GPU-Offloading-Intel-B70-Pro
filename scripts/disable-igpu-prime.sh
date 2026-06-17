#!/bin/bash
# Remove the iGPU-PRIME "Sonderbehandlung" so the desktop simply uses
# whichever GPU is attached to the monitor (no forced primary).
#
# The original setup (for FluidX3D, to free B70 VRAM) forced the desktop
# onto the integrated GPU via FOUR pieces; the OGL/OpenFOAM work has shown
# the iGPU passthrough causes more problems than it solves, so we revert it:
#
#   1. /etc/default/grub                          -> drop "desktop_gpu=igpu" kernel flag, timeout 5 -> 3
#   2. /etc/grub.d/11_desktop_b70                 -> remove the extra "desktop_gpu=b70" menu entry
#   3. /etc/udev/rules.d/61-mutter-primary-igpu.rules -> remove the "iGPU = mutter preferred primary" tag
#   4. /etc/X11/xorg.conf.d/20-igpu-display.conf  -> remove the "force X11 display onto iGPU (PCI:0:2:0)" config
#
# KEPT (NOT touched -- not display/PRIME related):
#   intel_iommu=igfx_off        (fixes DMA-remap + GEM-BO leak on B70)
#   transparent_hugepage=always (memory perf tweak)
#   If you also want these gone, edit /etc/default/grub by hand afterwards.
#
# Everything is backed up first; a restore script is written next to the backup.
# Run as root:   sudo bash scripts/disable-igpu-prime.sh
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: must run as root.  ->  sudo bash $0" >&2
    exit 1
fi

TS=$(date +%Y%m%d-%H%M%S)
BK="/root/igpu-prime-backup-${TS}"
mkdir -p "$BK"
echo "[*] Backups -> $BK"

GRUB=/etc/default/grub
GRUBD=/etc/grub.d/11_desktop_b70
UDEV=/etc/udev/rules.d/61-mutter-primary-igpu.rules
XORG=/etc/X11/xorg.conf.d/20-igpu-display.conf

# --- back up whatever exists ---
for f in "$GRUB" "$GRUBD" "$UDEV" "$XORG"; do
    [[ -e "$f" ]] && cp -a "$f" "$BK/" && echo "    backed up $f"
done

# --- write a restore script ---
cat > "$BK/restore.sh" <<RESTORE
#!/bin/bash
# Undo disable-igpu-prime.sh -- restores the iGPU-PRIME setup from this backup.
set -e
[[ \$EUID -ne 0 ]] && { echo "run as root: sudo bash \$0" >&2; exit 1; }
cp -a "$BK/grub" "$GRUB" 2>/dev/null && echo "restored $GRUB" || true
cp -a "$BK/11_desktop_b70" "$GRUBD" 2>/dev/null && echo "restored $GRUBD" || true
cp -a "$BK/61-mutter-primary-igpu.rules" "$UDEV" 2>/dev/null && echo "restored $UDEV" || true
cp -a "$BK/20-igpu-display.conf" "$XORG" 2>/dev/null && echo "restored $XORG" || true
update-grub
echo "Restored. Reboot to apply."
RESTORE
chmod +x "$BK/restore.sh"
echo "[*] Restore script -> $BK/restore.sh"

# --- 1. /etc/default/grub : drop desktop_gpu=igpu, set timeout 3 ---
echo "[*] Editing $GRUB"
sed -i -E 's/ +desktop_gpu=[a-z0-9]+//g' "$GRUB"
sed -i -E 's/^GRUB_TIMEOUT=.*/GRUB_TIMEOUT=3/' "$GRUB"
echo "    new cmdline:  $(grep '^GRUB_CMDLINE_LINUX_DEFAULT=' "$GRUB")"
echo "    new timeout:  $(grep '^GRUB_TIMEOUT=' "$GRUB")"

# --- 2. remove custom grub menu entry ---
if [[ -e "$GRUBD" ]]; then rm -f "$GRUBD"; echo "[*] Removed $GRUBD"; fi

# --- 3. remove udev mutter-primary rule ---
if [[ -e "$UDEV" ]]; then rm -f "$UDEV"; echo "[*] Removed $UDEV"; fi

# --- 4. remove X11 iGPU-display config ---
if [[ -e "$XORG" ]]; then rm -f "$XORG"; echo "[*] Removed $XORG"; fi

# --- regenerate grub.cfg ---
echo "[*] Running update-grub"
update-grub

echo
echo "DONE. The desktop will use the monitor-attached GPU after the next reboot."
echo "To undo:  sudo bash $BK/restore.sh"
