# HotSpot 6.0 with all-layer grid dump

HotSpot 6.0 ([uvahotspot/HotSpot](https://github.com/uvahotspot/HotSpot) commit `47b1e83`, line endings converted to
LF) with `layer-dump.patch` applied, as used in HotSpot-HMS (Hong et al., HPCA 2024): `-grid_steady_file <f>` writes the
steady grid of every layer to `<f>[i]` instead of the top layer only. License: `LICENSE` (University of Virginia).

Build (done by `python3 -m thermal build`):

```
make -C thermal/hotspot hotspot SUPERLU=0 EXTRAFLAGS=-fcommon
```

`-fcommon` is needed with GCC 10 or newer; Apple clang accepts it. SuperLU stays off.
