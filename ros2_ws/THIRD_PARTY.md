# Third-party content and licensing

This workspace is **not uniformly MIT**. It ships, and derives from, DexHand
content that is CC BY-NC-SA 4.0. Read this before using any of it.

> This is a description of what is here and what the upstream licences say. It
> is not legal advice. If this repository is going to be used for anything
> beyond research, have whoever handles licensing at the lab confirm it.

## What is under which licence

| Path | Licence | Origin |
| --- | --- | --- |
| `src/**` (code, configs, launch) | MIT, as the parent repository | Written here |
| `src/berkeley_humanoid_lite_description/urdf/*.urdf` | see below | Generated |
| `src/berkeley_humanoid_lite_description/meshes_dexhand_left/` | **CC BY-NC-SA 4.0** | Derived: mirrored from `dexhandv2_description` |
| `vendor/dexhandv2_description/` | **CC BY-NC-SA 4.0** | https://github.com/iotdesignshop/dexhandv2_description |
| `vendor/dexhand_v1_description/` | **CC BY-NC-SA 4.0** | https://github.com/iotdesignshop/dexhand_description |
| `src/berkeley_humanoid_lite_description/meshes` (symlink) | MIT | The parent repo's own asset submodule |

Each vendored directory keeps its upstream `LICENSE` and a `VENDORED.md`
recording the exact commit it came from.

## What CC BY-NC-SA 4.0 requires

- **BY** — attribution. Preserved via the `LICENSE` and `VENDORED.md` files.
- **NC** — no commercial use. This covers the vendored meshes and anything
  derived from them, so the hand geometry in this repository is research-only.
- **SA** — share alike. Derivative works carry the same licence. That is why
  `meshes_dexhand_left/` is marked CC BY-NC-SA and not MIT: mirroring a mesh
  produces a derivative of it.

## The generated URDFs sit in between

`berkeley_humanoid_lite_dexhand.urdf`, `_v1arm.urdf` and `_tuning.urdf` are
assembled by `generate_urdf.py` from three sources:

- the parent repo's humanoid URDF (MIT),
- the DexHand V2 description (CC BY-NC-SA), and
- the DexHand V1 description's wrist joint origins, axes and limits, plus its
  forearm and wrist inertials (CC BY-NC-SA).

They embed values taken from CC BY-NC-SA files and reference CC BY-NC-SA meshes,
so treat them as carrying that licence too. `berkeley_humanoid_lite.urdf` (the
`stock` model) contains no DexHand content and stays MIT.

## Keeping it separable

None of the DexHand content is mixed into the MIT source tree. The vendored
material lives entirely under `vendor/`, the one derived artefact is its own
directory, and both are reproducible:

```bash
./fetch_vendor.sh                                                  # full upstream clones
python3 src/berkeley_humanoid_lite_description/mirror_meshes.py    # regenerate the left hand
```

Deleting `vendor/` and `meshes_dexhand_left/` leaves a tree with no CC BY-NC-SA
content in it; the `stock` model still builds and runs from that state.
