#!/usr/bin/bash
set -e

python train_ae.py \
	--data_dir=data \
	--batch_size=15 \
	--epochs=3 \
	--lr=2e-2 \
	--val_size=0.2 \
	--val_batch_size=32 \
	--val_interval=1 \
	--results_dir=results \
	--device=cpu \

python see_params.py > out.txt
