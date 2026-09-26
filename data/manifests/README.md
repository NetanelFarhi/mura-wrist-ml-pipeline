# Local manifest policy

`data/manifests/` is created locally by the audit and split scripts. Full manifests are intentionally ignored because they enumerate a protected local MURA installation and are not required for another authorized user to recreate them.

This repository may track only this policy file and deliberately reviewed aggregate summaries that contain no image paths, patient identifiers, credentials, or raw image data. An authorized user recreates the full manifests locally from their own MURA copy by following [`../README.md`](../README.md).
