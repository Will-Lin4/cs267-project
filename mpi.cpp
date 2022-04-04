#include "common.h"
#include <mpi.h>
#include <map>

void sparse_all_reduce(const int num_procs, const int rank,
                       const int vector_len,
                       const std::map<int, int>& in_vector,
                       std::map<int, int>& reduced_vector) {
}
