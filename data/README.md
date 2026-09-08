# Processed simulation dataset

`safe_col_mix.txt` has 14,715 whitespace-delimited rows, no header, eight columns, and 981 scenario IDs. Each scenario has 15 frames. Sort by scenario ID and then frame index; do not reorder the generated prefix samples independently of their saved split keys.

| Zero-based column | Name | Meaning |
|---|---|---|
| 0 | frame | Integer frame index within the scenario, 0 through 14 |
| 1 | distance | Ego-to-target separation in raw simulation distance units |
| 2 | angle | Relative bearing, radians |
| 3 | sin_yaw | Sine of the supplied relative motion-direction angle |
| 4 | cos_yaw | Cosine of the supplied relative motion-direction angle |
| 5 | speed | Relative-motion magnitude in raw distance units per second; not absolute target speed |
| 6 | label | Observed PU endpoint label s: 1 is a labeled positive, 0 is unlabeled |
| 7 | sequence_id | Integer scenario ID |

The manuscript and CMPA code use 9 raw distance units = 0.9 m at the physical QCar scale. Multiply raw distance by 0.1 to express that physical-scale distance; applying the same scale to raw distance/second gives physical-scale m/s. Model inputs are retained exactly as supplied and are not converted before TF evaluation. The stated sample interval is 0.135 s (approximately 7.4 Hz). Historical acquisition code uses elapsed processing time for some speed estimates; the original acquisition timestamps and complete preprocessing provenance were not supplied.

The comparison models interpret `x = distance*cos(angle)`, `y = distance*sin(angle)`, `vx = speed*cos_yaw`, `vy = speed*sin_yaw`. The archived raw feature extractor uses an atan2 argument order different from the manuscript equation. This release documents that unresolved upstream coordinate convention and does not silently rotate or regenerate the supplied features.

The supplied code derives scenario outcome as `max(label)` within each sequence. This gives 371 positive-outcome and 610 non-collision scenarios. It is a derived outcome, not a separate independently supplied event-annotation table. Observed positives number 3,424 rows. The paper describes positive censoring at distance 9 in collision scenarios; the script that originally assigned labels was not supplied.

Main evaluation forms prefixes of lengths 2 through 15, labeled by their last frame: 13,734 samples. Common GAT/CMPA/Su-LSTM comparisons use endpoints 3 through 14 (lengths 4 through 15): 11,772 samples. Current-risk evaluation measures agreement with observed s, whereas Su-LSTM training uses the derived scenario outcome Y. The corresponding outcome evaluation is reported separately.
