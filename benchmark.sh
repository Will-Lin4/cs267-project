#!/bin/bash

# confirm overwriting $output_file if already exists
output_file='benchmark.out'
if [ -f $output_file ]; then
    read -p "File $output_file already exists. Overwrite? [y/N] " overwrite
	if [[ $overwrite == [yY] ]]; then
		rm $output_file
	else
		exit 1
	fi
fi

# VARY VECTOR LENGTH
# max_pow=24
# num_trials=3
# for i in $(seq 1 $max_pow); do
# 	count="[$i / $max_pow]"
# 	n=$(echo "2^$i" | bc)
# 	#cmd="srun -n 256 ./mpi -d uniform -r 1729 -s 0.01 -n $n"
# 	cmd="srun -n 64 ./mpi -d uniform -r 1729 -s 0.001 -n $n"

# 	echo "$count $cmd" | tee -a $output_file
# 	for i in $(seq 1 $num_trials); do
# 		output=$(eval $cmd)
# 		echo "$output" | tail -1 >> $output_file
# 	done
# 	printf '\n' >> $output_file
# done

# VARY PROCESSOR COUNT (uniform + geometric distribution)
num_trials=3
processors=( 1 2 4 8 16 32 48 64 96 128 160 192 256 )
for i in ${!processors[@]}; do
	count="[$((i+1)) / ${#processors[@]}]"
	cmd="srun -n ${processors[$i]} ./mpi -d geometric -r 1729 -n 16777216 -s 0.001 -p 0.0001"
	#cmd="srun -n ${processors[$i]} ./mpi -d uniform -r 1729 -n 16777216 -s 0.001"

	echo "$count $cmd" | tee -a $output_file
	for i in $(seq 1 $num_trials); do
		output=$(eval $cmd)
		echo "$output" | tail -1 >> $output_file
	done
	printf '\n' >> $output_file
done

# VARY DENSITY
# num_trials=3
# sparsity=( 0.0001 0.0002 0.0004 0.0008 0.0012 0.0016 0.0024 0.0032 0.0064 0.0128 0.0256 0.0512 0.1024 0.2048 )
# for i in ${!sparsity[@]}; do
# 	count="[$((i+1)) / ${#sparsity[@]}]"
# 	cmd="srun -n 64 ./mpi -r 1729 -d uniform -n 16777216 -s ${sparsity[$i]}"

# 	echo "$count $cmd" | tee -a $output_file
# 	for i in $(seq 1 $num_trials); do
# 		output=$(eval $cmd)
# 		echo "$output" | tail -1 >> $output_file
# 	done
# 	printf '\n' >> $output_file
# done

# VARY SKEW
# num_trials=1
# #p=( 0.0000000001 0.000000001 0.00000001 0.0000001 0.000001 0.00001 0.0001 0.001 )
# p=( 0.000000001 0.00000001 0.0000001 0.000001 0.00001 0.0001 )
# for i in ${!p[@]}; do
# 	count="[$((i+1)) / ${#p[@]}]"
# 
# 	cmd="srun -N 4 --ntasks-per-node=64 ./mpi -r 1729 -n 10000000 -s 0.01 -d geometric -p ${p[$i]}"
# 
# 	echo "$count $cmd" | tee -a $output_file
# 	for i in $(seq 1 $num_trials); do
# 		output=$(eval $cmd)
# 		echo "$output" | tail -1 >> $output_file
# 	done
# 	printf '\n' >> $output_file
# done
