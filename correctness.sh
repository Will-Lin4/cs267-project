#!/bin/bash

extra=1 # duplicate commands for every distribution
output_file='correctness_checks.out'
commands=(
	'srun ./mpi'
	'srun -n 1 ./mpi -n 0'
	'srun -n 5 ./mpi -n 1'
	'srun -n 4 ./mpi -n 2'
	'srun -n 4 ./mpi -n 4'
	'srun -n 7 ./mpi -n 133'
	'srun -n 42 ./mpi -n 10'
	'srun -n 10 ./mpi -n 100'
	'srun -n 13 ./mpi -n 10000'
	'srun -n 17 ./mpi -n 1000000'
	'srun -n 68 ./mpi -n 1000000'
)

# confirm overwriting $output_file if already exists
if [ -f $output_file ]; then
    read -p "File $output_file already exists. Overwrite? [y/N] " overwrite
	if [[ $overwrite == [yY] ]]; then
		rm $output_file
	else
		exit 1
	fi
fi

# for each command which doesn't specify distribution
# add two copies for geometric and poisson distributions
if [[ $extra -eq 1 ]]; then
	for cmd in "${commands[@]}"; do
		if [[ $cmd != *"-d"* ]]; then
			vector_len=$(echo "$cmd" | awk -F'./mpi' '{print $2}' | awk -F'-n ' '{print $2}' | cut -d ' ' -f1)
			if [[ $vector_len -gt 30 ]]; then
				sparsity=$(echo "scale=10 ; 10 / $vector_len" | bc)
			else
				sparsity=0.3
			fi
			commands+=( "$cmd -d geometric -s $sparsity" )
			commands+=( "$cmd -d poisson -s $sparsity" )
		fi
	done
fi

# run all commands, output to file, notify if incorrect
num_fail=0
for i in ${!commands[@]}; do
	count="[$((i+1)) / ${#commands[@]}]"
	cmd="${commands[$i]} -c"

	echo "$count $cmd"
	echo "$count $cmd" >> $output_file
	output=$(eval $cmd)
	echo "$output" >> $output_file
	printf '\n' >> $output_file

	error=$(echo $output | grep 'Allreduce Time')
	if [[ -z "$error" ]]; then
		echo "↑↑↑ ERROR (try again) ↑↑↑"
		num_fail=$((num_fail + 1))
		continue
	fi

	incorrect=$(echo $output | grep -i 'incorrect')
	if [[ ! -z "${incorrect// }" ]]; then
		echo "↑↑↑ INCORRECT RESULT ↑↑↑"
		num_fail=$((num_fail + 1))
		continue
	fi
done

# output results
echo "========================================"
if [[ $num_fail -eq 0 ]]; then
	echo "PASSED ALL TESTS SUCCESSFULLY!"
else
	echo "FAILED $num_fail / ${#commands[@]} TESTS!"
fi