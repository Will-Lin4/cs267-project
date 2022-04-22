#!/bin/bash

output_file='benchmark_naive.out'
#output_file='benchmark_sparse.out'
command='srun -n 68 ./mpi -s 0.01'

#output_file='benchmark_generic.out'
#output_file='benchmark_distribution.out'

# confirm overwriting $output_file if already exists
if [ -f $output_file ]; then
    read -p "File $output_file already exists. Overwrite? [y/N] " overwrite
	if [[ $overwrite == [yY] ]]; then
		rm $output_file
	else
		exit 1
	fi
fi

# naive vs. sparse
max_pow=24
num_trials=5
for i in $(seq 1 $max_pow); do
	count="[$i / $max_pow]"
	n=$(echo "2^$i" | bc)

	if [[ $output_file == "benchmark_naive.out" ]]; then
		cmd="$command -n $n -naive"
	else
		cmd="$command -n $n"
	fi

	echo "$count $cmd"
	echo "$count $cmd" >> $output_file
	for i in $(seq 1 $num_trials); do
		output=$(eval $cmd)
		echo "$output" | tail -1 >> $output_file
	done
	printf '\n' >> $output_file
done

# # generic vs. distribution
# num_trials=5
# processors=( 1 2 4 8 12 16 24 32 40 48 54 62 68 )
# for i in ${!processors[@]}; do
# 	count="[$((i+1)) / ${#processors[@]}]"
# 	cmd="srun -n ${processors[$i]} ./mpi -d geometric -r 1729 -n 10000000 -s 0.001 -p 0.0001"

# 	echo "$count $cmd"
# 	echo "$count $cmd" >> $output_file
# 	for i in $(seq 1 $num_trials); do
# 		output=$(eval $cmd)
# 		echo "$output" | tail -1 >> $output_file
# 	done
# 	printf '\n' >> $output_file
# done
