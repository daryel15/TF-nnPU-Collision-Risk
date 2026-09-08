# Inputs not supplied with the original archive

The runnable processed-feature release has its required inputs. These separate historical assets remain unavailable:

- The file originally called `safe_col_mix_dth_9.txt`; byte identity with `data/safe_col_mix.txt` cannot be established. Cost-sensitive reruns explicitly use the released dataset and are labeled new runs.
- Weights originally called `pretrained_encoder.pt` and `pretrained_encoder_good.pt`; their identity cannot be established. Clean workflows explicitly use fresh pretraining or the named released encoder.
- Matched predictions and RNG states for the historical cost-sensitive significance table. New reruns now export them; historical aggregate numbers remain reference-only.
- Physical QCar features, collision annotations, metadata and frame maps, including the input folder referenced by the historical full notebook.
- Raw camera/LiDAR recordings, YOLO detector weights, complete QLabs scene/control code, original acquisition timestamps, and the script that assembled/labeled the processed text file.

These limitations are not repaired by renaming a different dataset or claiming newly trained models are historical artifacts. The original workspace remains unchanged. Current and prior manuscript PDFs and the response letter are retained in the author's original folder rather than bundled in this code release; their inclusion is not necessary to run the software. Historical scientific source/adaptation notes are retained in the archive.
