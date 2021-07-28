#!/usr/bin/bash
set -e

Ls="16 32 64 128"

for L in $Ls
do
	python train_ae.py \
		--data_dir=data \
		--L=$L \
		--n_train_val=2500 \
		--n_test=5000 \
		--batch_size=128 \
		--epochs=10 \
		--lr=2e-3 \
		--val_size=0.5 \
		--val_batch_size=512 \
		--val_interval=1 \
		--results_dir=results \
		--observable_name=latent_symmetric \
		--device=cpu \
		--symmetric
done

echo All done!
