# Phase 1 XR_WRIST EDA and data-quality report

> **Current governance note:** “sealed” below records the historical Phase-1 split protocol. Official MURA `valid` was later observed once for the frozen Phase-1 reference and is not untouched for subsequent selection; see [TEST_SET_GOVERNANCE.md](TEST_SET_GOVERNANCE.md).

## Scope and evidence

This report covers only Stanford MURA `XR_WRIST` records. Counts were recomputed from local-only wrist and study-level split manifests (excluded from Git), then checked against [wrist_data_audit.json](wrist_data_audit.json). The recomputed totals agree with the audit: **3,474 patients, 3,697 studies, and 10,411 images**.

MURA label `0` denotes a NORMAL study and label `1` an ABNORMAL study. This is an abnormality-classification dataset; these values do not establish fracture diagnoses or locations.

## Dataset counts and label balance

| Unit / label | Count | Share within unit |
|---|---:|---:|
| Patients | 3,474 | — |
| Studies | 3,697 | — |
| Images | 10,411 | — |
| NORMAL studies | 2,274 | 61.51% of studies |
| ABNORMAL studies | 1,423 | 38.49% of studies |
| NORMAL images | 6,129 | 58.87% of images |
| ABNORMAL images | 4,282 | 41.13% of images |

The official source composition is 3,460 studies / 9,752 images from `train` and 237 studies / 659 images from official `valid`. The manifest maps the entire official `valid` source to the sealed final `test` split.

## Patient-safe final splits

| Final split | Patients | Studies | Images | NORMAL studies | ABNORMAL studies | ABNORMAL-study share |
|---|---:|---:|---:|---:|---:|---:|
| Train | 2,614 | 2,768 | 7,805 | 1,707 | 1,061 | 38.33% |
| Validation | 653 | 692 | 1,947 | 427 | 265 | 38.29% |
| Test (official `valid`, sealed) | 207 | 237 | 659 | 140 | 97 | 40.93% |
| **Total** | **3,474** | **3,697** | **10,411** | **2,274** | **1,423** | **38.49%** |

The train/validation partition was created from official training patients with seed `23` and a requested validation fraction of `0.20`. The resulting study and image proportions are close to 75% / 19% / 6%; the unit of separation is the patient, so exact image-level proportions are not the selection criterion.

## Views per study

The prediction unit remains a study. Studies contain 1–8 images, with a mean of **2.816** images per study.

| Images (views) in study | Studies | Share of studies |
|---:|---:|---:|
| 1 | 300 | 8.11% |
| 2 | 705 | 19.07% |
| 3 | 2,138 | 57.83% |
| 4 | 509 | 13.77% |
| 5 | 28 | 0.76% |
| 6 | 10 | 0.27% |
| 7 | 5 | 0.14% |
| 8 | 2 | 0.05% |

Most studies have three views. Single-view studies are retained rather than removed, and padded study batches must use a view mask so padding cannot affect aggregation.

## Image dimensions

All **10,411** manifest images were successfully inspected. There are 464 observed width-by-height combinations. Width ranges from **131 to 512** pixels and height from **223 to 512** pixels.

| Dimension | Images | Share of images |
|---|---:|---:|
| 406 × 512 | 2,240 | 21.52% |
| 379 × 512 | 957 | 9.19% |
| 407 × 512 | 679 | 6.52% |
| 512 × 406 | 557 | 5.35% |
| 405 × 512 | 537 | 5.16% |

By orientation, 9,330 images (89.62%) are portrait, 900 (8.64%) are landscape, and 181 (1.74%) are square. The preprocessing pipeline therefore preserves aspect ratio with square padding before resize; it does not assume a single native image shape.

## Integrity and leakage checks

| Check | Finding |
|---|---:|
| Duplicate image paths | 0 |
| Missing files | 0 |
| Corrupt/unreadable images | 0 |
| Inconsistent labels within a study | 0 |
| Manifest label versus folder-label mismatches | 0 |
| Study mapped to multiple patients | 0 |
| Patient overlap across final splits | 0 |
| Study overlap across final splits | 0 |
| Unassigned manifest studies | 0 |

The audited manifest is internally consistent and the official validation source is isolated as a sealed test set. These are data-pipeline findings only: no model was trained, selected, or evaluated as part of this report, and no model-performance metrics are claimed.
