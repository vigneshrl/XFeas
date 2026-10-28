#!/bin/bash
# Fetch the released controllers / planners XFeas runs on top of, at the exact commits we used. Nothing here is modified:
# XFeas only sends each controller its own commands (velocity / height / pitch / roll / gait).
#   bash third_party/setup.sh            # everything
#   bash third_party/setup.sh AMO relic  # only some
# Afterwards also unpack the XFeas extras (robot USDs converted from each controller's MJCF/URDF, the GR00T-WBC policy
# folder, the VAMOS adapter) into third_party/ -- see README, "Setup".
set -e
cd "$(dirname "$(readlink -f "$0")")"

# name                    url                                                     commit
REPOS="
AMO                       https://github.com/OpenTeleVision/AMO.git                34caaf9
GR00T-WholeBodyControl    https://github.com/NVlabs/GR00T-WholeBodyControl.git    45c03fa
php_parkour               https://github.com/amazon-far/php_parkour.git            c7c9ce3
relic                     https://github.com/rai-opensource/relic.git              27f8033
mppi_numba                https://github.com/mit-acl/mppi_numba.git                1bbb22f
vamos                     https://github.com/vamos-vla/vamos.git                   bfe078b
PegasusSimulator          https://github.com/PegasusSimulator/PegasusSimulator.git bef3c57
"


WANT="$*"

echo "$REPOS" | while read -r name url commit; do
  [ -z "$name" ] && continue
  if [ -n "$WANT" ] && [[ " $WANT " != *" $name "* ]]; then continue; fi
  if [ ! -d "$name/.git" ]; then git clone "$url" "$name"; fi
  git -C "$name" fetch --quiet origin && git -C "$name" checkout --quiet "$commit"
  echo "[third_party] $name @ $commit"
done

# GEAR-SONIC (SONIC, the held-out G1 controller) is released as a Hugging Face model, not a git repo
if [ -z "$WANT" ] || [[ " $WANT " == *" GEAR-SONIC "* ]]; then
  if [ ! -f GEAR-SONIC/config.json ]; then
    python -m pip install --quiet huggingface_hub
    huggingface-cli download nvidia/GEAR-SONIC --local-dir GEAR-SONIC
  fi
  echo "[third_party] GEAR-SONIC (huggingface.co/nvidia/GEAR-SONIC)"
fi
