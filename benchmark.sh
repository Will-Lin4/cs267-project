#!/bin/bash

# INSTRUCTIONS:
# 0. For each run below, do it three times for:
#    (1) dense (2) sparse (3) dynamic allgather [lines 527-532 in mpi.cpp]
# 1. Run VARY PROCESSOR COUNT section for each of three command ("cmd") types
#    [line 47: geometric, 1% density]
#    [line 48: geometric, 50% density]
#    [line 49: uniform, 1% density]
# 2. Run VARY DENSITY section

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
# num_trials=5
# command='srun -n 68 ./mpi -s 0.01'
# for i in $(seq 1 $max_pow); do
# 	count="[$i / $max_pow]"
# 	n=$(echo "2^$i" | bc)

# 	if [[ $output_file == "benchmark_naive.out" ]]; then
# 		cmd="$command -n $n -naive"
# 	else
# 		cmd="$command -n $n"
# 	fi

# 	echo "$count $cmd"
# 	echo "$count $cmd" >> $output_file
# 	for i in $(seq 1 $num_trials); do
# 		output=$(eval $cmd)
# 		echo "$output" | tail -1 >> $output_file
# 	done
# 	printf '\n' >> $output_file
# done

# VARY PROCESSOR COUNT (uniform + geometric distribution)
num_trials=3
processors=( 1 2 4 8 12 16 24 32 40 48 54 62 68 )
for i in ${!processors[@]}; do
	count="[$((i+1)) / ${#processors[@]}]"

	#cmd="srun -n ${processors[$i]} ./mpi -d geometric -r 1729 -n 16777216 -s 0.01 -p 0.00001"
	cmd="srun -n ${processors[$i]} ./mpi -d geometric -r 1729 -n 524288 -s 0.5 -p 0.00001"
	#cmd="srun -n ${processors[$i]} ./mpi -d uniform -r 1729 -n 16777216 -s 0.01"

	echo "$count $cmd"
	echo "$count $cmd" >> $output_file
	for i in $(seq 1 $num_trials); do
		output=$(eval $cmd)
		echo "$output" | tail -1 >> $output_file
	done
	printf '\n' >> $output_file
done

# VARY DENSITY
# num_trials=5
# sparsity=( 0.0002 0.0004 0.0008 0.0012 0.0016 0.0024 0.0032 0.0064 0.0128 )
# for i in ${!sparsity[@]}; do
# 	count="[$((i+1)) / ${#sparsity[@]}]"
# 	cmd="srun -n 256 ./mpi -r 1729 -d uniform -n 16777216 -s ${sparsity[$i]}"

# 	echo "$count $cmd"
# 	echo "$count $cmd" >> $output_file
# 	for i in $(seq 1 $num_trials); do
# 		output=$(eval $cmd)
# 		echo "$output" | tail -1 >> $output_file
# 	done
# 	printf '\n' >> $output_file
# done
