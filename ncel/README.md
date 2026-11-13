# NorthCel sources

Every drawing in the film is an editable `.ncel` file.

```
sources/
  cels/f0001…f0090.ncel    the 90 source drawings
  extras/m*.ncel           the in-between drawings
  start_face.ncel          the starting drawing
  receipts/*.json          fit receipts
film/
  film.ncel                the plan: keys and their transitions
  receipt.json             plan receipt and audio source
authored/
  a*.ncel                  vector in-between blocks
v7/
  film.ncel                the complete 300-frame, 20 fps film
```

All `.ncel` files pass the published validator:

```sh
find . -name '*.ncel' -print0 | xargs -0 -n1 northcel check
```

`v7/film.ncel` is written by `process/v7/build.py`; see the top-level
[README](../README.md).
