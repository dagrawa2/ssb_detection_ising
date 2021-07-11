#!/usr/bin/bash
set -e

python train_ae.py \
	--data_dir=data/isingdata \
	--batch_size=64 \
	--epochs=10 \
	--lr=2e-3 \
	--val_size=0.2 \
	--val_batch_size=500 \
	--val_interval=5 \
	--results_dir=results/AE \
	--device=cpu \
