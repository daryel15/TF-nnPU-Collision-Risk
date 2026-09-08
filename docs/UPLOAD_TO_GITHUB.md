# Publish this release

1. Create a public GitHub repository, for example `TF-nnPU-Collision-Risk`.
2. Upload the contents of the release folder, including its directories, so README.md is at the repository root. Upload the files in batches if the browser limits the number per upload; do not use the ZIP as the only source artifact.
3. Include `.gitignore` and the `.github` directory when using GitHub Desktop or Git. These may be hidden in file pickers.
4. Commit the files. Open the public repository in a signed-out browser and check access.
5. Create a version `v0.1.0` release and optionally attach the distribution ZIP.
6. Add the actual repository URL to CITATION.cff and the manuscript's availability statement. Regenerate MANIFEST_SHA256.txt after tracked-file edits using `python tools/update_manifest.py`.

Suggested availability wording: “The processed simulation dataset, released encoder and collision-risk checkpoint, evaluation notebook, and training workflows are available at [actual repository URL]. This release supports processed-feature evaluation and documented experiment reruns; unavailable historical and physical-data assets are identified in the repository.”

Repository creation and publication are performed through the user's GitHub account; this local preparation does not publish anything automatically.
