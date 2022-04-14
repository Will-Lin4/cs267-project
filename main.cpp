#include "common.h"
#include <chrono>
#include <random>

// =================
// Helper Functions
// =================

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
int generate_vector(std::map<int, int>& vector, const int vector_len, const int num_procs, const int rank,
					const int random_seed, const double sparsity, const char* distribution, const double dist_param) {
	std::mt19937 tmp(random_seed);
	for (int i = 0; i < rank; i++) tmp(); // so that different processors get different starting vectors
	std::mt19937 gen(tmp());

	std::uniform_int_distribution<> uniform(0, dist_param-1);
	std::geometric_distribution<> geometric(dist_param);
	std::poisson_distribution<> poisson(dist_param);

	int nz_count = 0, nz_max = std::round(sparsity * vector_len);
	while (nz_count < nz_max) {
		int idx;
		if (!strcmp(distribution, "uniform")) idx = uniform(gen);
		else if (!strcmp(distribution, "geometric")) idx = geometric(gen);
		else if (!strcmp(distribution, "poisson")) idx = poisson(gen);
		else { std::cerr << "Distribution '" << distribution << "' not found\n"; return -1; }

		if (!vector.count(idx)) {
			vector.emplace(idx, 1);
			nz_count++;
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

// ==============
// Main Function
// ==============

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
			std::cout << "-v: verbose output" << std::endl;
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
	int vector_len = find_int_arg(argc, argv, "-n", 1000);
	double sparsity = find_double_arg(argc, argv, "-s", 0.3);
	char* distribution = find_string_arg(argc, argv, "-d", (char*) "uniform");
	double dist_param = find_distribution_parameter(argc, argv, vector_len, distribution);

	std::map<int, int> in_vector;
	std::map<int, int> reduced_vector;

	// Generate input vector
	if (generate_vector(in_vector, vector_len, num_procs, rank, random_seed, sparsity, distribution, dist_param))
		return -1;

	// Algorithm
	auto start_time = std::chrono::steady_clock::now();
	sparse_all_reduce(num_procs, rank, vector_len, in_vector, reduced_vector, distribution, dist_param);
	auto end_time = std::chrono::steady_clock::now();

	// Output results
	if (find_arg_idx(argc, argv, "-v") >= 0) {
		display_sparse_vector(in_vector, vector_len);
		if (rank == num_procs-1) {
			for (int i = 0; i < 2*vector_len + 3; i++) {
				std::cout << '-';
			}
			std::cout << '\n';
			display_sparse_vector(reduced_vector, vector_len);
			std::cout << '\n';
		}
	}
	if (rank == num_procs-1) {
		std::chrono::duration<double> diff = end_time - start_time;
		std::cout << "Allreduce Time: " << diff.count() << " sec\n";
	}

	MPI_Finalize();
}