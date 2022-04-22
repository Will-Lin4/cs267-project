#!/bin/bash

#output_file='benchmark_naive.out'
#output_file='benchmark_sparse.out'
#command='srun -n 68 ./mpi -r 1729 -s 0.01'

output_file='benchmark_generic.out'
#output_file='benchmark_distribution.out'
command='srun -n 10 ./mpi -r 1729 -d geometric'

# confirm overwriting $output_file if already exists
if [ -f $output_file ]; then
    read -p "File $output_file already exists. Overwrite? [y/N] " overwrite
	if [[ $overwrite == [yY] ]]; then
		rm $output_file
	else
		exit 1
	fi
fi

# # naive vs. sparse
# max_pow=24
# num_trials=5
# for i in $(seq 1 $max_pow); do
# 	count="[$i / $max_pow]"
# 	n=$(echo "2^$i" | bc)

# 	if [[ $output_file == "benchmark_naive.out" ]]; then
# 		cmd="$command -n $n"
# 	else
# 		cmd="$command -n $n -naive"
# 	fi

# 	echo "$count $cmd"
# 	echo "$count $cmd" >> $output_file
# 	for i in $(seq 1 $num_trials); do
# 		output=$(eval $cmd)
# 		echo "$output" | tail -1 >> $output_file
# 	done
# 	printf '\n' >> $output_file
# done

# generic vs. distribution
n=100
num_trials=5
for i in $(seq 1 10); do
	count="[$i / 10]"
	p=$(echo "scale=3 ; $i/10" | bc)
	s=$(echo "scale=3 ; (11-$i) / $n" | bc)
	cmd="$command -n $n -p $p -s $s"

	echo "$count $cmd"
	echo "$count $cmd" >> $output_file
	for i in $(seq 1 $num_trials); do
		output=$(eval $cmd)
		echo "$output" | tail -1 >> $output_file
	done
	printf '\n' >> $output_file
done
