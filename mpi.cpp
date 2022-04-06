#include "common.h"
#include <mpi.h>
#include <map>
#include <vector>

void sparse_all_reduce(const int num_procs, const int rank,
                       const int vector_len,
                       const std::map<int, int>& in_vector,
                       std::map<int, int>& reduced_vector) {
}

/* Populates a vector with the contents of a map */
void map_to_vector(const std::map<int, int>& m, const std::vector<int>& v) {

    for (auto i = m.begin(); i != m.end(); ++i) {
        v[i->first] = i->second; 
    }

}

/* Populates a map with the contents of an array */
void array_to_map(int *v, const std::map<int, int>& m) {
    
    for (int i = 0; i < m.size(); ++i) {
        m[i] = v[i];
    }

}

/* Copies data to another vector, used at the start of direct allreduce to populate reduced vector with local data */
void copy_vector(const std::map<int, int>& in_vector, const std::map<int, int>& reduced_vector) {


    for (auto i = in_vector.begin(); i != in_vector.end(); ++i) {
        reduced_vector[i->first] = i->second;
    }

}

/* sums local_data and recv_data, writing the result to local_data */
void operate(const std::map<int, int>& local_data, const std::map<int, int>& recv_data) {

    for (auto i = recv_data.begin(); i != recv_data.end(); ++i) {
        local_data[i->first] += i->second;
    }

}

void trade(const int num_procs, const int rank,
                       const int vector_len,
                       std::map<int, int>& reduced_vector) {
    int trade_distance = num_procs/2;

    std::vector<int> local_data;
    local_data.resize(vector_len);
    map_to_vector(reduced_vector, local_data);

    std::map<int, int> recv_map;
    
    int *recv_data = new int[vector_len]; 
    MPI_Request *requests = new MPI_Request[2];
    MPI_Request r1;
    MPI_Request r2;
    requests[0] = r1;
    requests[1] = r2;

    while(trade_distance >= 1) {
        /* Send and receive vectors */
        if (rank>=num_procs/2) {
            MPI_Isend(local_data.data(), vector_len, MPI_INT, rank-trade_distance, 
                    NULL, MPI_COMM_WORLD, &requests[0]);
            MPI_Irecv(recv_data, vector_len, MPI_INT, rank-trade_distance, 
                    NULL, MPI_COMM_WORLD, &requests[1]);
            MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
        } else { 
            MPI_Isend(local_data.data(), vector_len, MPI_INT, rank+trade_distance, 
                    NULL, MPI_COMM_WORLD, &requests[0]);
            MPI_Irecv(recv_data, vector_len, MPI_INT, rank+trade_distance, 
                    NULL, MPI_COMM_WORLD, &requests[1]);
            MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
        }

        /* Perform operation on local and received vecotrs */
        array_to_map(recv_data, recv_map);
        operate(reduced_vector, recv_map);
        memset(recv_data, 0, sizeof(int)*vector_len);
        trade_distance -= 1;
    }
}

void collect(const int num_procs, const int rank,
                       const int vector_len,
                       const std::map<int, int>& in_vector,
                       std::map<int, int>& reduced_vector) {
    
}

void distribute(const int num_procs, const int rank,
                       const int vector_len,
                       const std::map<int, int>& in_vector,
                       std::map<int, int>& reduced_vector) {
    
}

void direct_all_reduce(const int num_procs, const int rank,
                       const int vector_len,
                       const std::map<int, int>& in_vector,
                       std::map<int, int>& reduced_vector) {
    // Copy, trade, collect, distribute
}