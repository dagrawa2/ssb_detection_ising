#!/usr/bin/bash
set -e

Js=("ferromagnetic" "antiferromagnetic")
Ls=(16 32 64 128)
Ns=(8 16 32 64 128 256)

Es=(32 32 32 32 32 32)
Bs=(8 16 32 64 128 256)
LR=1e-3

scales=("16,32" "16,32,64,128")
nscales=(2 4)

fold=$1
seeds=(0 1 2)

for J in ${Js[@]}
do
	for L in ${Ls[@]}
	do
		for i in ${!Ns[@]}
		do
			for seed in ${seeds[@]}
			do
				# baseline-autoencoder
				python train.py \
					--data_dir=data/$J \
					--L=$L \
					--fold=$fold \
					--seed=$seed \
					--n_train_val=${Ns[i]} \
					--n_test=2048 \
					--batch_size=${Bs[i]} \
					--epochs=${Es[i]} \
					--lr=$LR \
					--val_size=0.5 \
					--val_batch_size=2048 \
					--val_interval=1 \
					--results_dir=results3/$J/latent/L$L/N${Ns[i]}/fold$fold/seed$seed \
					--device=cpu
				#
				# GE-autoencoder
				python train.py \
					--data_dir=data/$J \
					--L=$L \
					--fold=$fold \
					--seed=$seed \
					--n_train_val=${Ns[i]} \
					--n_test=2048 \
					--batch_size=${Bs[i]} \
					--epochs=${Es[i]} \
					--lr=$LR \
					--val_size=0.5 \
					--val_batch_size=2048 \
					--val_interval=1 \
					--results_dir=results3/$J/latent_equivariant/L$L/N${Ns[i]}/fold$fold/seed$seed \
					--device=cpu \
					--symmetric \
					--equivariance_reg=1.0 \
					--equivariance_pre=$((Es[i]/2))
			done
		done
	done
done


# multiscale GE-autoencoder
for seed in ${seeds[@]}
do
	for J in ${Js[@]}
	do
		for i in ${!Ns[@]}
		do
			for j in ${!scales[@]}
			do
				python train.py \
					--data_dir=data/$J \
					--Ls=${scales[j]} \
					--L=16 \
					--L_test=16 \
					--fold=$fold \
					--seed=$seed \
					--n_train_val=${Ns[i]} \
					--n_test=2048 \
					--batch_size=${Bs[i]} \
					--epochs=${Es[i]} \
					--lr=$LR \
					--val_size=0.5 \
					--val_batch_size=2048 \
					--val_interval=1 \
					--results_dir=results3/$J/latent_multiscale_${nscales[j]}/L16/N${Ns[i]}/fold$fold/seed$seed \
					--device=cpu \
					--symmetric \
					--equivariance_reg=1.0 \
					--equivariance_pre=$((Es[i]/2))
			done
		done
	done
done

# evaluate 2-scale GE-autoencoder on other lattice sizes
Ls=(32 64 128)
for seed in ${seeds[@]}
do
	for J in ${Js[@]}
	do
		for L in ${Ls[@]}
		do
			for i in ${!Ns[@]}
			do
				python train.py \
					--data_dir=data/$J \
					--L=$L \
					--fold=$fold \
					--seed=$seed \
					--n_train_val=2 \
					--n_test=2048 \
					--batch_size=${Bs[i]} \
					--epochs=0 \
					--lr=$LR \
					--val_size=0.5 \
					--val_batch_size=2048 \
					--val_interval=1 \
					--load_model=results3/$J/latent_multiscale_2/L16/N${Ns[i]}/fold$fold/seed$seed \
					--results_dir=results3/$J/latent_multiscale_2/L$L/N${Ns[i]}/fold$fold/seed$seed \
					--device=cpu \
					--symmetric \
					--equivariance_reg=1.0 \
					--equivariance_pre=$((Es[i]/2))
			done
		done
	done
done

# evaluate 4-scale GE-autoencoder on other lattice sizes
Ls=(32 64 128)
for seed in ${seeds[@]}
do
	for J in ${Js[@]}
	do
		for L in ${Ls[@]}
		do
			for i in ${!Ns[@]}
			do
				python train.py \
					--data_dir=data/$J \
					--L=$L \
					--fold=$fold \
					--seed=$seed \
					--n_train_val=2 \
					--n_test=2048 \
					--batch_size=${Bs[i]} \
					--epochs=0 \
					--lr=$LR \
					--val_size=0.5 \
					--val_batch_size=2048 \
					--val_interval=1 \
					--load_model=results3/$J/latent_multiscale_4/L16/N${Ns[i]}/fold$fold/seed$seed \
					--results_dir=results3/$J/latent_multiscale_4/L$L/N${Ns[i]}/fold$fold/seed$seed \
					--device=cpu \
					--symmetric \
					--equivariance_reg=1.0 \
					--equivariance_pre=$((Es[i]/2))
			done
		done
	done
done


echo All done!
