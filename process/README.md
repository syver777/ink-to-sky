# Process

| Path | Purpose |
| --- | --- |
| `polish/palette.json` | the 17-colour shared palette (`[[r,g,b], …]`) used to trace every drawing |
| `polish/config.json` | cut, morph and dissolve choices and the match thresholds for `northnoir polish` |
| `polish/extra_keys.json` | the in-between drawings: `{id, a, b, src, u}` with 1-based cel numbers |
| `polish/reviews.json` | the reviewer verdict for each in-between drawing |
| `v6/authoring.json` | which drawing each film frame comes from |
| `v7/build.py` | builds `ncel/v7/film.ncel`, renders the frames and encodes the film |
| `v7/cloud.py` | places the character with the cloud drawn in each key |
| `v7/face.py` | eye style and in-between scale normalisation |

See the top-level [README](../README.md) for the commands.
