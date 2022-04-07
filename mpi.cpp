#include "common.h"
#include <mpi.h>
#include <map>
#include <vector>

void sparse_all_reduce(const int num_procs, const int rank,
                       const int vector_len,
                       const std::map<int, int>& in_vector,
                       std::map<int, int>& reduced_vector) {
}


/*
* HELPER FUNCTIONS
*/

/* Populates a vector with the contents of a map */
void map_to_vector(std::map<int, int>& m, std::vector<int>& v) {

    for (auto i = m.begin(); i != m.end(); ++i) {
        v[i->first] = i->second; 
    }

}

/* Populates a map with the contents of an array */
void array_to_map(int *v,  std::map<int, int>& m) {
    
    for (int i = 0; i < m.size(); ++i) {
        m[i] = v[i];
    }

}

/* Copies data to another vector, used at the start of direct allreduce to populate reduced vector with local data */
void copy_vector(std::map<int, int>& in_vector, std::map<int, int>& reduced_vector) {


    for (auto i = in_vector.begin(); i != in_vector.end(); ++i) {
        reduced_vector[i->first] = i->second;
    }

}

/* sums local_data and recv_data, writing the result to local_data */
void operate(std::map<int, int>& local_data, std::map<int, int>& recv_data) {

    for (auto i = recv_data.begin(); i != recv_data.end(); ++i) {
        local_data[i->first] += i->second;
    }

}

/* Halves the input map */

void halve(std::map<int, int> m, bool odd) {
    int limit = m.size()/2;
    int cnt = 0;
    if (odd) { //first half 
        for (std::map<int, int>::iterator i = m.begin(); cnt < limit; ++i) {
            m.erase(i);
            cnt++;
        }
    } else { //second half
        for (std::map<int, int>::iterator i = m.begin(); i != m.end(); ++i) {
            if (cnt>=limit)
                m.erase(i);
        }
    }
}

/* Direct allreduce */

/* Distance doubling. 1/n of the local buffer to another processor */
void distance_double_vector_halve(std::map<int, int>& data, const int dist, const int rank, const int num_procs, const int n) {
    MPI_Request *requests = new MPI_Request[2];
    MPI_Request r1;
    MPI_Request r2;
    requests[0] = r1;
    requests[1] = r2;

    std::map<int, int> recv_map;
    
    int *recv_data = new int[n]; 

    std::vector<int> send_data;
    vector.resize(n);
    map_to_vector(data, send_data);

    if (rank%2) { //even rank

        MPI_Isend(send_data.data(), n, MPI_INT, rank+dist, NULL, MPI_COMM_WORLD, &requests[0]);
        MPI_Irecv(recv_data, n, MPI_INT, rank-dist, NULL, MPI_COMM_WORLD, &requests[1]);
        MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
        
    } else { //odd rank

        MPI_Isend(send_data.data(), n, MPI_INT, rank-dist, NULL, MPI_COMM_WORLD, &requests[0]);
        MPI_Irecv(recv_data, n, MPI_INT, rank+dist, NULL, MPI_COMM_WORLD, &requests[1]);
        MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

    }

    //Operate on received data
    array_to_map(recv_data, recv_map);
    operate(data, recv_map);

    delete recv_data;
}

/* Distance halving */
void distance_halve_vector_double(std::map<int, int>& data, const int dist, const int rank, const int num_procs, const int n) {
    MPI_Request *requests = new MPI_Request[2];
    MPI_Request r1;
    MPI_Request r2;
    requests[0] = r1;
    requests[1] = r2;

    std::vector<int> send_data;
    vector.resize(n);
    map_to_vector(data, send_data);

    
}

void direct_all_reduce(const int num_procs, const int rank,
                       const int vector_len,
                       const std::map<int, int>& in_vector,
                       std::map<int, int>& reduced_vector) {

    int dist = 1;
    int n = vector_len/2;

    //distance doubling and vector halving step
    while (dist<=num_procs/2) {
        halve(reduced_vector, rank%2);
        distance_double_vector_halve(reduced_vector, dist, rank, num_procs, n);
        dist *= 2;
        n = n / 2;
    }

    //vector doubling and distance halving


}