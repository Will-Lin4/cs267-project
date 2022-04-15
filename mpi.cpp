#include "common.h"
#include <cstring>
#include <iostream>
#include <math.h>
#include <mpi.h>
#include <vector>

#define TAG_REDUCE_SCATTER 0
#define TAG_ALL_GATHER 1
#define EPSILON 0 // TODO: determine optimal value for epsilon

// =================
// Helper Functions
// =================

/* Modulus operator without a negative output */
long factorial(const int n) {
    long f = 1;
    for (int i = 2; i <= n; i++)
        f *= i;
    return f;
}

double pmf (const int x, const char* distribution, const double dist_param) {
    if (!strcmp(distribution, "uniform")) {
        return 1/(double)dist_param;
    } else if (!strcmp(distribution, "geometric")) {
        return pow(1-dist_param, x-1) * dist_param;
    } else if (!strcmp(distribution, "poisson")) {
        return pow(dist_param, x) * exp(-dist_param) / factorial(x);
    }

    return (double)-1;
}

/* Partitions {0,...,vector_len-1} into num_procs subsets of roughly equal mass
   according to the probability mass function specified by the distribution */
std::vector<int> compute_partition_chunks(
        const int num_procs, int &num_active_procs, const int my_rank,
        const int vector_len, const char* distribution,
        const double dist_param) {

    double total_mass = 0;
    for (int i = 0; i < vector_len; i++) {
        total_mass += pmf(i, distribution, dist_param);
    }

    int rank = 1;
    double mass = 0;
    const double uniform_mass = 1/(double)num_procs;
    std::vector<int> chunk_boundaries; // chunks[i] is first index of processor i's chunk
    chunk_boundaries.push_back(0);
    for (int idx = 0; idx < vector_len - 1; idx++) {
        /* Assign indices to chunk until cumulative probability mass
           is at least within EPSILON of (ideal) uniform mass */
        mass += pmf(idx, distribution, dist_param) / total_mass;
        if (mass >= uniform_mass - EPSILON) {
            chunk_boundaries.push_back(idx + 1);
            rank++;
            mass = 0;
        }

        if (rank == num_procs)
            break; // stop after last processor's chunk is determined
    }

    chunk_boundaries.push_back(vector_len); // for convenience, so chunks[i+1] always exists
    num_active_procs = rank;

    // if (my_rank == 0) {
    //     std::cout << "( ";
    //     for (int i = 0; i < chunks.size(); i++) {
    //         std::cout << chunks[i] << ' ';
    //     }
    //     std::cout << ")\n";
    // }

    return std::move(chunk_boundaries);
}

// =================
// Direct Allreduce
// =================

//void unpack(MPI_Request& request, int* buffer, std::map<int, int> reduced_chunk) {
//    int recv_size;
//    MPI_Get_count(&recv_status, MPI_INT, &recv_size);
//
//    for (int i = 0; i < recv_size; i += 2) {
//        int idx = recv_data[i];
//        int val = recv_data[i+1];
//
//        reduced_chunk.emplace(idx, 0);
//        reduced_chunk[idx] += val;
//    }
//}

std::map<int, int> reduce_scatter(const int num_procs, const int num_active_procs,
                                  const int my_rank, const int vector_len,
                                  const std::map<int, int>& in_vector,
                                  const std::vector<int> chunk_boundaries) {
    std::vector<MPI_Request> all_requests;
    all_requests.reserve(num_active_procs * 2); // Both send and recv

    // Scatter: For each active processor, send it the relevant chunk of in_vector
    std::vector<std::vector<int>> all_send_data;
    all_send_data.reserve(num_active_procs);

    for (int rank = 0; rank < num_active_procs; rank++) {
        if (rank == my_rank)
            continue;

        // Setup send data: [idx_1, val_1, ..., idx_nnz, val_nnz]
        all_send_data.emplace_back();
        auto& send_data = all_send_data.back();

        for (auto it = in_vector.lower_bound(chunk_boundaries[rank]);
             it != in_vector.upper_bound(chunk_boundaries[rank + 1]); it++) {
            send_data.push_back(it->first);
            send_data.push_back(it->second);
        }

        all_requests.emplace_back(MPI_REQUEST_NULL);
        MPI_Isend(send_data.data(), send_data.size(), MPI_INT, rank,
                  TAG_REDUCE_SCATTER, MPI_COMM_WORLD, &all_requests.back());
    }

    size_t num_sends = all_requests.size();

    // Reduce: Receive and reduce all chunks from other processors
    std::map<int, int> reduced_chunk;
    if (my_rank < num_active_procs) {
        // Initialize reduced_chunk with in_vector
        for (auto it = in_vector.lower_bound(chunk_boundaries[my_rank]);
                it != in_vector.upper_bound(chunk_boundaries[my_rank + 1]); it++) {
            reduced_chunk.emplace(it->first, it->second);
        }

        int* recv_buffer = new int[2 * vector_len];
        for (int rank = 0; rank < num_procs; rank++) {
            if (rank == my_rank)
                continue;

            MPI_Status recv_status;
            MPI_Recv(recv_buffer, 2 * vector_len, MPI_INT, rank,
                     TAG_REDUCE_SCATTER, MPI_COMM_WORLD, &recv_status);

            int recv_size;
            MPI_Get_count(&recv_status, MPI_INT, &recv_size);

            for (int i = 0; i < recv_size; i += 2) {
                int idx = recv_buffer[i];
                int val = recv_buffer[i+1];

                reduced_chunk.emplace(idx, 0);
                reduced_chunk[idx] += val;
            }
        }

        delete recv_buffer;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    return std::move(reduced_chunk);
}

void all_gather(const int num_procs, const int num_active_procs,
                const int my_rank, const int vector_len,
                const std::map<int, int>& reduced_chunk,
                std::map<int, int>& reduced_vector) {

    std::vector<MPI_Request> all_requests;
    all_requests.reserve(num_active_procs * 2); // Both send and recv

    // Send reduced_chunk to all other processors
    std::vector<int> send_data; 
    if (my_rank < num_active_procs) {
        for (const auto& elem : reduced_chunk) {
            reduced_vector.emplace(elem.first, elem.second);

            send_data.push_back(elem.first);
            send_data.push_back(elem.second);
        }

        for (int rank = 0; rank < num_procs; rank++) {
            if (rank == my_rank) {
                continue;
            }

            all_requests.emplace_back(MPI_REQUEST_NULL);
            MPI_Isend(send_data.data(), send_data.size(), MPI_INT, rank,
                      TAG_ALL_GATHER, MPI_COMM_WORLD, &all_requests.back());
        }
    }

    // Read reduced chunks from all active processors
    int* recv_data = new int[2 * vector_len];
    for (int rank = 0; rank < num_active_procs; rank++) {
        if (rank == my_rank) {
            continue;
        }

        MPI_Status recv_status;
        MPI_Recv(recv_data, 2 * vector_len, MPI_INT, rank, TAG_ALL_GATHER,
                 MPI_COMM_WORLD, &recv_status);

        int recv_size;
        MPI_Get_count(&recv_status, MPI_INT, &recv_size);
        for (int i = 0; i < recv_size; i += 2) {
            reduced_vector.emplace(recv_data[i], recv_data[i+1]);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    delete recv_data;
}

void dist_sparse_all_reduce(const int num_procs, const int rank,
                            const int vector_len,
                            const std::map<int, int>& in_vector, 
                            const char* distribution, const double dist_param,
                            std::map<int, int>& reduced_vector) {
    int num_active_procs;
    std::vector<int> chunk_boundaries = 
        compute_partition_chunks(num_procs, num_active_procs, rank, vector_len,
                                 distribution, dist_param);

    std::map<int, int> reduced_chunk =
        reduce_scatter(num_procs, num_active_procs, rank, vector_len,
                       in_vector, chunk_boundaries);

    all_gather(num_procs, num_active_procs, rank, vector_len, reduced_chunk, reduced_vector);
    MPI_Barrier(MPI_COMM_WORLD);
}
