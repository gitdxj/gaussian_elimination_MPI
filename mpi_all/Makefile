block_mpi, loop_mpi, pipe_mpi, omp_mpi, avx_mpi, sse_mpi:sse_main.cpp avx_main.cpp block_main.cpp loop_main.cpp pipe_main.cpp omp_main.cpp matrix.cpp
	mpic++ -march=corei7-avx -o sse_mpi sse_main.cpp matrix.cpp
	mpic++ -march=corei7-avx -o avx_mpi avx_main.cpp matrix.cpp
	mpic++ -march=corei7-avx -o block_mpi block_main.cpp matrix.cpp
	mpic++ -march=corei7-avx -o loop_mpi loop_main.cpp matrix.cpp
	mpic++ -march=corei7-avx -o pipe_mpi pipe_main.cpp matrix.cpp
	mpic++ -march=corei7-avx -o omp_mpi omp_main.cpp matrix.cpp
	
clean:
	rm *_mpi