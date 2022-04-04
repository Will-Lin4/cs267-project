#include "common.h"
#include <chrono>
#include <cstring>
#include <iostream>
#include <map>
#include <mpi.h>

// =================
// Helper Functions
// =================

// Command Line Option Processing
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

char* find_string_option(int argc, char** argv, const char* option, char* default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1) {
        return argv[iplace + 1];
    }

    return default_value;
}

// Initializes a vector with the values [rank + 1, rank + 1, ..., rank + 1]
void init_dummy_vector(const int num_procs, const int rank,
                       const int vector_len, std::map<int, int>& in_vector) {
    for (int i = 0; i < vector_len; i++) {
        in_vector.emplace(i, rank + 1);
    }
}

bool test_dummy_reduced_vector(const int num_procs,
                               const int vector_len,
                               const std::map<int, int>& reduced_vector) {
    int expected_val = num_procs * (num_procs + 1) / 2;
    if (reduced_vector.size() != vector_len) {
        return false;
    }

    int i = 0;
    for (const auto& elem_it : reduced_vector) {
        if (elem_it.first != i) {
            return false;
        }

        if (elem_it.second != expected_val) {
            return false;
        }

        i++;
    }

    return true;
}

// ==============
// Main Function
// ==============

int main(int argc, char** argv) {
    // Parse Args
    if (find_arg_idx(argc, argv, "-h") >= 0) {
        std::cout << "Options:" << std::endl;
        std::cout << "-h: see this help" << std::endl;
        std::cout << "-n <int>: set vector length" << std::endl;
        return 0;
    }

    // Init MPI
    int num_procs, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int vector_len = find_int_arg(argc, argv, "-n", 1000);

    std::map<int, int> in_vector;
    std::map<int, int> reduced_vector;

    // TODO: Replace with an actual generator
    init_dummy_vector(num_procs, rank, vector_len, in_vector);
    
    // Algorithm
    auto start_time = std::chrono::steady_clock::now();
    sparse_all_reduce(num_procs, rank, vector_len, in_vector, reduced_vector);
    auto end_time = std::chrono::steady_clock::now();

    bool correct = test_dummy_reduced_vector(num_procs, vector_len,
                                             reduced_vector);
    if (!correct) {
        std::cout << "Incorrect result on Rank: <" << rank << ">" << std::endl; 
    } else if (rank == 0) {
        std::chrono::duration<double> diff = end_time - start_time;
        double seconds = diff.count();

        std::cout << "Length: <" << vector_len << ">"
                  << ", "
                  << "Time: <" << seconds << "s>" << std::endl; 
    }

    MPI_Finalize();
}
