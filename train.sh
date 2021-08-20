#!/usr/bin/bash
set -e

Ls="16 32 64 128"
Js="ferromagnetic antiferromagnetic"

for J in $Js
do
	for L in $Ls
	do
		python train.py \
			--data_dir=data/$J \
			--L=$L \
			--n_train_val=2500 \
			--n_test=5000 \
			--batch_size=128 \
			--epochs=5 \
			--lr=2e-3 \
			--val_size=0.5 \
			--val_batch_size=2048 \
			--val_interval=1 \
			--results_dir=results/$J \
			--observable_name=latent_equivariant \
			--device=cpu \
			--symmetric \
			--equivariance_reg=1.0 \
			--equivariance_pre=2
	done
done

for J in $Js
do
	for L in $Ls
	do
		python train.py \
			--data_dir=data/$J \
			--L=$L \
			--n_train_val=2500 \
			--n_test=5000 \
			--batch_size=128 \
			--epochs=5 \
			--lr=2e-3 \
			--val_size=0.5 \
			--val_batch_size=2048 \
			--val_interval=1 \
			--results_dir=results/$J \
			--observable_name=latent \
			--device=cpu
	done
done

echo All done!
