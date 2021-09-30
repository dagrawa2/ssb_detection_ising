#!/usr/bin/bash
set -e

Js="ferromagnetic antiferromagnetic"
Ls="16 32 64 128"
Ns="16 32 64 128 256 512 1024 2048"

for J in $Js
do
	for L in $Ls
	do
		for N in $Ns
		do
			python train.py \
				--data_dir=data/$J \
				--L=$L \
				--n_train_val=$N \
				--n_test=5000 \
				--batch_size=128 \
				--epochs=5 \
				--lr=2e-3 \
				--val_size=0.5 \
				--val_batch_size=2048 \
				--val_interval=1 \
				--results_dir=results_new/$J \
				--observable_name=latent_equivariant \
				--device=cpu \
				--symmetric \
				--equivariance_reg=1.0 \
				--equivariance_pre=2
		done
	done
done

for J in $Js
do
	for L in $Ls
	do
		for N in $Ns
		do
			python train.py \
				--data_dir=data/$J \
				--L=$L \
				--n_train_val=$N \
				--n_test=5000 \
				--batch_size=128 \
				--epochs=5 \
				--lr=2e-3 \
				--val_size=0.5 \
				--val_batch_size=2048 \
				--val_interval=1 \
				--results_dir=results_new/$J \
				--observable_name=latent \
				--device=cpu
		done
	done
done

echo All done!
