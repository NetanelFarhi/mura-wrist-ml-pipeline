# Phase 2C training analysis

The partial model selected epoch 1 by validation AUROC from 5 completed epochs. Training and validation curves are in `training_curves.png`. For the partial run, train loss moved from 0.3903 to 0.2029, while validation loss moved from 0.4742 to 0.5384 and validation AUROC moved from 0.913767 to 0.908570. This is an overfitting signal after the first epoch, so the first-epoch checkpoint was retained. The selected checkpoint was determined from internal validation only; no official MURA `valid` row was loaded.
