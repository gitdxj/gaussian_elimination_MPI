# gaussian_elimination_MPI  
并行程序设计课的一次作业  
这次作业满分是8分得了7.2  
估计是因为没有做pthread  
# 作业要求
1.实现数据划分的算法：块划分，负载不均；（块）循环划分，负载均衡  
2.与pthread（openmp）、sse（avx）结合  
（选做）：实现流水线算法  
流水线算法写了好像也不加分hhhhh  
# 文件结构
+ mpi_lu_block  块划分  
+ mpi_lu_loop  循环划分  
+ mpi_lu_pipeline  流水线算法  
+ mpi_lu_pipeline_omp	 流水线算法+OpenMP  
+ mpi_lu_pipeline_sse&avx  流水线算法+SSE和AVX  
# 编译运行
make编译  
这个MakeFile写的很烂  
在个人电脑上可以用mpiexec在本地运行：  
***  
**mpiexec  -n  &lt;number of processes&gt;  &lt;executable&gt;**
***
