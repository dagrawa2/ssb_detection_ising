#!/usr/bin/bash
set -e

python train_ae.py \
	--data_dir=data \
	--batch_size=16 \
	--epochs=50 \
	--lr=5e-2 \
	--val_size=0.2 \
	--val_batch_size=64 \
	--val_interval=10 \
	--results_dir=results_ae \
	--device=cpu \
