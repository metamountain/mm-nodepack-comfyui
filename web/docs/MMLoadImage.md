# Load Image MM (`MMLoadImage`)

Load one image, or work through a folder as a batch. Give the node a file (upload, drag & drop, paste or a full path) or a folder, and step through the batch in file order or at random. Preview & Save MM finds this node automatically as its source.

## How to use

1. Set `mode`: `single` for one image, `folder` for a batch.
2. In `single` mode: pick a file in the OS dialog, upload one, or type a full file path into `path`. `path` wins over the uploaded image.
3. In `folder` mode: set `folder` (or use the choose-folder control). The images load in filename order.
4. Choose `incremental` to step through the folder with `index`, or `random` to let `index` seed the pick.
5. Wire `image` to your downstream nodes; for a batch, wire `image list`.
6. Queue the workflow. In single mode the `index` control sits on `fixed`.

## Inputs / Outputs

- **Inputs:** `mode`, `image`, `path`, `folder`, `batch_mode`, `index`
- **Outputs:** `image`, `mask` (inverted alpha), `width`, `height`, `path`, `image list`, `index`

## Tips

- `path` always wins over the uploaded image; a missing file in single mode is an error.
- `image list` is always in filename order (`img_2` before `img_10`), also in random mode; each image keeps its own resolution.
- `index` is 1-based and wraps after the last image.
- The file dialogs each remember their own last folder (open, save and folder pick are separate).
- With a browser on another computer, picking a file falls back to an upload into the `input` folder.

## Shared server

On a shared server — or a remote browser session on a private one — this node reads only from the `input` and `output` folders and writes only to `output`. Opening files from the OS dialog and saving next to the source image are available to local users. The operator switches the policy on with the server's listen address (anything other than `127.0.0.1`) or the `MM_NODEPACK_SHARED_SERVER` variable.
