#include "common.h"
#include <chrono>
#include <cstring>
#include <iostream>
#include <random>
#include <mpi.h>

/* Command Line Option Processing */
int find_arg_idx(int argc, char** argv, const char* option) {
	for (int i = 1; i < argc; ++i) {
		if (strcmp(argv[i], option) == 0) {
			return i;
		}
	}
	return -1;
}

int find_int_arg(int argc, char** argv, const char* option, int default_value) {
	int iplace = find_arg_idx(argc, argv, option);

	if (iplace >= 0 && iplace < argc - 1) {
		return std::stoi(argv[iplace + 1]);
	}

	return default_value;
}

double find_double_arg(int argc, char** argv, const char* option, double default_value) {
	int iplace = find_arg_idx(argc, argv, option);

	if (iplace >= 0 && iplace < argc - 1) {
		return std::stof(argv[iplace + 1]);
	}

	return default_value;
}

char* find_string_arg(int argc, char** argv, const char* option, char* default_value) {
	int iplace = find_arg_idx(argc, argv, option);

	if (iplace >= 0 && iplace < argc - 1) {
		return argv[iplace + 1];
	}

	return default_value;
}

double find_distribution_parameter(int argc, char** argv, const int vector_len, const char* distribution) {
	double dist_param = find_double_arg(argc, argv, "-p", -1);

	if (!strcmp(distribution, "uniform")) {
		if (dist_param < 1) dist_param = vector_len;
	} else if (!strcmp(distribution, "geometric")) {
		if (dist_param < 0 || dist_param > 1) dist_param = 0.5;
	} else if (!strcmp(distribution, "poisson")) {
		if (dist_param < 0) dist_param = 4;
	}

	return dist_param;
}

/* Generates sparse vector according to specified sparsity and distribution */
int generate_vector(const int vector_len, const int num_procs, const int rank,
					const int random_seed, const double sparsity,
					const char* distribution, const double dist_param,
					std::map<int, int>& vector) {
	std::uniform_int_distribution<> uniform(0, dist_param-1);
	std::geometric_distribution<> geometric(dist_param);
	std::poisson_distribution<> poisson(dist_param);

	std::mt19937 gen(random_seed * 1000 + rank);
	std::uniform_int_distribution<> value_generator(0, 1024);

	int geometric_start = 0;
	int nz_count = 0, nz_target = std::round(sparsity * vector_len);
	while (nz_count < nz_target) {
		int idx;
		if (!strcmp(distribution, "uniform")) idx = uniform(gen);
		else if (!strcmp(distribution, "geometric")) idx = geometric_start + geometric(gen);
		else if (!strcmp(distribution, "poisson")) idx = poisson(gen);
		else { std::cerr << "Distribution '" << distribution << "' not found\n"; return -1; }

		if (idx < vector_len && !vector.count(idx)) {
			vector.emplace(idx, value_generator(gen));
			nz_count++;

			if (!strcmp(distribution, "geometric")) {
				while (vector.count(geometric_start)) {
					geometric_start += 1;
				}
			}
		}
	}

	return 0;
}

/* Displays dense represenation of sparse vector */
void display_sparse_vector(std::map<int, int>& vector, const int vector_len) {
	std::cout << "[ ";
	for (int i = 0; i < vector_len; i++) {
		if (vector.count(i)) {
			std::cout << vector[i] << ' ';
		} else {
			std::cout << "0 ";
		}
	}
	std::cout << "]\n";
}

/* Displays dense vector */
void display_dense_vector(int* vector, const int vector_len) {
	std::cout << "[ ";
	for (int i = 0; i < vector_len; i++) {
		std::cout << vector[i] << ' ';
	}
	std::cout << "]\n";
}

/* Returns true iff reduced_vector and correct_reduced_vector have the same elements */
bool is_correct(int* reduced_vector, int* correct_reduced_vector, const int vector_len) {
	for (int i = 0; i < vector_len; i++) {
		if (reduced_vector[i] != correct_reduced_vector[i]) {
			return false;
		}
	}
	return true;
}

int main(int argc, char** argv) {
	// Init MPI
	int num_procs, rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// Parse Args
	if (find_arg_idx(argc, argv, "-h") >= 0) {
		if (rank == 0) {
			std::cout << "Options:" << std::endl;
			std::cout << "-h: see this help" << std::endl;
			std::cout << "-c: perform correctness checks" << std::endl;
			std::cout << "-r <int>: set random seed" << std::endl;
			std::cout << "-n <int>: set vector length" << std::endl;
			std::cout << "-s <double>: set sparsity" << std::endl;
			std::cout << "-d <string>: set distribution (UNIFORM, geometric, poisson)" << std::endl;
			std::cout << "-p <double>: set distribution parameter" << std::endl;
		}
		return 0;
	}

	std::random_device rd;
	int random_seed = find_int_arg(argc, argv, "-r", rd());
	int vector_len = find_int_arg(argc, argv, "-n", 10);
	double sparsity = find_double_arg(argc, argv, "-s", 0.3);
	char* distribution = find_string_arg(argc, argv, "-d", (char*) "uniform");
	double dist_param = find_distribution_parameter(argc, argv, vector_len, distribution);

	std::map<int, int> in_vector;
	int* reduced_vector = new int[vector_len];

	// Generate input vector
	if (generate_vector(vector_len, num_procs, rank, random_seed,
						sparsity, distribution, dist_param, in_vector))
		return -1;

	if (rank == 0) {
		std::cout << "-----------------------------" << "\n"
				  << "Distribution: " << distribution << " (" << dist_param << ")\n"
				  << "Length: " << vector_len << "\n"
				  << "Sparsity: " << sparsity << "\n"
				  << "-----------------------------" << "\n";
	}

	// Perform Allreduce
	auto start_time = std::chrono::steady_clock::now();
	if (find_arg_idx(argc, argv, "-d") >= 0)
		dist_sparse_all_reduce(num_procs, rank, vector_len, in_vector,
							   distribution, dist_param, reduced_vector);
	else
		dist_sparse_all_reduce(num_procs, rank, vector_len, in_vector,
							   "unknown", dist_param, reduced_vector);
	MPI_Barrier(MPI_COMM_WORLD);
	auto end_time = std::chrono::steady_clock::now();

	// Output results
	if (find_arg_idx(argc, argv, "-c") >= 0) {
		if (rank == 0) {
			std::cout << "Checking correctness..." << std::endl;
		}

		int* correct_reduced_vector = new int[vector_len];
		naive_sparse_all_reduce(num_procs, rank, vector_len, in_vector, correct_reduced_vector);

		// if (rank == 0) {
		// 	display_dense_vector(reduced_vector, vector_len);
		// 	display_dense_vector(correct_reduced_vector, vector_len);
		// }

		if (!is_correct(reduced_vector, correct_reduced_vector, vector_len)) {
			std::cout << "Incorrect result on Rank: <" << rank << ">" << std::endl;
		}
	}

	if (rank == 0) {
		std::chrono::duration<double> diff = end_time - start_time;
		std::cout << "Allreduce Time: " << diff.count() << " sec\n";
	}

	MPI_Finalize();
}
