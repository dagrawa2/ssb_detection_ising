#!/usr/bin/bash
set -e

Js=("ferromagnetic" "antiferromagnetic")
Ls=(16 32 64 128)
Ns=(0 2 4 8 16 32 64 128 256 512 1024 2048)

Bs=(8 8 16 16 32 32 64 64 128 128 256 256)
Es=(256 256 128 128 64 64 32 32 16 16 8 8)

# GE-autoencoder
for J in ${Js[@]}
do
	for L in ${Ls[@]}
	do
		for i in ${!Ns[@]}
		do
			python train.py \
				--data_dir=data/$J \
				--L=$L \
				--n_train_val=${Ns[i]} \
				--n_test=5000 \
				--batch_size=${Bs[i]} \
				--epochs=${Es[i]} \
				--lr=2e-3 \
				--val_size=0.5 \
				--val_batch_size=2048 \
				--val_interval=1 \
				--results_dir=results/$J \
				--observable_name=latent_equivariant \
				--device=cpu \
				--symmetric \
				--equivariance_reg=1.0 \
				--equivariance_pre=$((Es[i]/2))
		done
	done
done


# baseline-autoencoder
for J in ${Js[@]}
do
	for L in ${Ls[@]}
	do
		for i in ${!Ns[@]}
		do
			python train.py \
				--data_dir=data/$J \
				--L=$L \
				--n_train_val=${Ns[i]} \
				--n_test=5000 \
				--batch_size=${Bs[i]} \
				--epochs=${Es[i]} \
				--lr=2e-3 \
				--val_size=0.5 \
				--val_batch_size=2048 \
				--val_interval=1 \
				--results_dir=results/$J \
				--observable_name=latent \
				--device=cpu
		done
	done
done


# reset sample sizes for multiscale observable
Ns=(8 16 32 64 128 256 512 1024 2048)

Bs=(16 32 32 64 64 128 128 256 256)
Es=(128 64 64 32 32 16 16 8 8)

# GE-autoencoder (multiscale)
for J in ${Js[@]}
do
	for L in ${Ls[@]}
	do
		for i in ${!Ns[@]}
		do
			python train.py \
				--data_dir=data/$J \
				--L=$L \
				--Ls=16,32,64,128 \
				--n_train_val=${Ns[i]} \
				--n_test=5000 \
				--batch_size=${Bs[i]} \
				--epochs=${Es[i]} \
				--lr=2e-3 \
				--val_size=0.5 \
				--val_batch_size=2048 \
				--val_interval=1 \
				--results_dir=results/$J \
				--observable_name=latent_multiscale \
				--device=cpu \
				--symmetric \
				--equivariance_reg=1.0 \
				--equivariance_pre=$((Es[i]/2))
		done
	done
done

echo All done!
