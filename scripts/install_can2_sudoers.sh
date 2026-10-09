#!/bin/bash
# One-time administrator installation. Does not configure or activate CAN.
set -euo pipefail
if [[ $EUID -ne 0 ]]; then
  echo 'Run this installer with sudo; it only installs the can2-specific rule.' >&2
  exit 1
fi
rule=/etc/sudoers.d/90-robot02-can2
scratch=$(mktemp /etc/sudoers.d/.robot02-can2.XXXXXX)
trap 'rm -f -- "$scratch"' EXIT
cat > "$scratch" <<'RULE'
# Robot approach: only exact can2 commands, no wildcard or shell privileges.
didi ALL=(root) NOPASSWD: /usr/sbin/ip link show can2, /usr/sbin/ip link set can2 type can bitrate 500000, /usr/sbin/ip link set can2 up, /usr/sbin/ip link set can2 down
RULE
chmod 0440 "$scratch"
/usr/sbin/visudo -cf "$scratch"
if [[ -e "$rule" ]]; then
  cmp -s "$scratch" "$rule" || { echo "Existing $rule differs; refusing to overwrite." >&2; exit 1; }
else
  /usr/bin/install -o root -g root -m 0440 "$scratch" "$rule"
  if ! /usr/sbin/visudo -c; then
    rm -f -- "$rule"
    echo 'Validation failed; new rule removed.' >&2
    exit 1
  fi
fi
echo 'Installed can2-only passwordless sudo. CAN state was not changed.'
