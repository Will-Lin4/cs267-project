#ifndef __CS267_COMMON_H__
#define __CS267_COMMON_H__

#include <iostream>
#include <mpi.h>
#include <map>

void sparse_all_reduce(const int num_procs, const int rank, const int vector_len,
                       const std::map<int, int>& in_vector, std::map<int, int>& reduced_vector);

#endif
