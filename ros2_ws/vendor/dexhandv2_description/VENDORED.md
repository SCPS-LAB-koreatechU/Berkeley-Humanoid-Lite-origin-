# Vendored: dexhandv2_description

Source: https://github.com/iotdesignshop/dexhandv2_description
Commit: e379b6ecad50
Licence: **CC BY-NC-SA 4.0** (see LICENSE in this directory)

V2 hand: the fingers this build uses, and the source of the mirrored left hand.

Only the files this workspace actually uses are kept; the upstream repository has
more. `../../fetch_vendor.sh` replaces this directory with a full clone if you
need the rest.

`config/dexhandv2_right_8servo.srdf` is **not** from the upstream repository. It
is the DexHand MoveIt configuration's own SRDF for the 8-servo build -- the 12
grip presets and the 194 intra-hand collision pairs its Setup Assistant computed
against real mesh geometry. `generate_srdf.py` used to read it from a path on
one developer's laptop and simply fail anywhere else; it is kept here so the
SRDFs regenerate from a fresh clone. Same origin, so the same licence applies.

This directory is **not** covered by the repository's MIT licence. CC BY-NC-SA
4.0 is NonCommercial and ShareAlike: it forbids commercial use and requires
derivative works to carry the same licence. See ../../THIRD_PARTY.md.
