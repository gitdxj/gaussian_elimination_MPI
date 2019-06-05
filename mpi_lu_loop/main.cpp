/*
实现块划分，设有m个mpi节点，方阵的规模为N*N
那么每一个节点分配N/m（这里的表示是不精确的）行的数据
设：n_ = (N-N%m)/m，对于rank号节点，其分配的行的范围为：
    rank != m-1时，[rank*_n, rank*_n+_n-1]
    rank == m-1时，[rank*_n, N-1]

*/


#include <mpi.h>
#include "matrix.h"
#include <stdio.h>
#include <sys/time.h>
#include <iostream>

using namespace std;
const int N = 1000;

void thread_perform(int rank);
void elimination(float **matrix/*矩阵*/, int n_row/*行数*/, float *row_k/*第k行*/, 
                int dimension/*每行维度*/, int k/*外层循环进行到k*/);
                void matrix_elimination(float **matrix, int n_row, float *row_k, int dimension, int k, int row_No);

int main(int argc, char ** argv)
{
    int rank;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    thread_perform(rank);
    
    MPI_Finalize();
    return 0;
}

void thread_perform(int rank)
{
	char *proc_name = new char[MPI_MAX_PROCESSOR_NAME];
	int name_len;
	MPI_Get_processor_name(proc_name, &name_len);

    int m = MPI::COMM_WORLD.Get_size();  // m是总的mpi节点个数
    int q = N % m;
    int _n = (N - q) / m;
    int matrix_size;  // 该节点所负责的行数
    if(rank <= q)
        matrix_size = _n + 1;
    else
        matrix_size = _n;
    
    float *row_k = new float[N];

	if(0 == rank)
	{
	
    	struct timeval start, end;
        // 因为矩阵中一行和线程是一一对应的关系，所以线程数得等于行数

		//  在rank号是0的线程内创建矩阵，并赋初值
		float **A = new float*[N];
		for(int i=0; i<N; i++)
			A[i] = new float[N];
		matrix_initialize(A, N, 100);

        float **A_copy = new float*[N];
        for(int i=0; i<N; i++)
			A_copy[i] = new float[N];

        copy_matrix(A_copy, A, N);
        gaussian_elimination_lu(A_copy, N);
		
		if(N <= 10)
			show_matrix(A, N);
        
        gettimeofday(&start, NULL);

        // 将各行的值发送给各线程
        for(int i = _n; i<N; i++)
        {
            // 计算第i行应该要发到的节点是哪个
            int dest = i / _n;
            if(m == dest)
                dest--;
            if(dest != 0)
                MPI_Send(A[i], N/*size*/, MPI_FLOAT/*type*/, dest/*dest*/, 0, MPI_COMM_WORLD);

        }

        // 对自己负责的行做division操作，发送结果，并对自己负责的行做消去
        for(int k=0; k < _n; k++)
        {
            // division
            for(int j=k+1; j<N; j++)
                A[k][j] = A[k][j] / A[k][k];
            A[k][k] = 1.0;

            // 将division结果发给后面的节点
            for(int dest=rank+1; dest<m; dest++){
                MPI_Send(A[k], N, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
            }

            // 对剩下的行做消去
            int rows_left = _n-1 - k;
            if(rows_left <= 0)
                break;
            
            matrix_elimination(A, _n, A[k], N, k, k+1);
            
        }

        for(int i=_n; i<N; i++)
        {
            // 确定第i行来自哪一个节点
            int src = i/_n;
            if(src == m)
                src--;
            MPI_Recv(A[i], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        
        
        
        gettimeofday(&end, NULL);
        unsigned long time_interval = 1000000*(end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
        //show_matrix(A, N);

        //cout << "standard lu: "<< endl;
        if(N <= 10)
			show_matrix(A, N);
        cout << "time is " << time_interval << endl;
        
        if(!ls_same(A, A_copy, N))
        	cout << "problem" << endl;
        else
        	cout << "correct" << endl;
        
	}
    else
    {
        // 用来存这一节点所负责的行
        float **mpi_matrix = new float*[matrix_size];
        for(int i=0; i<matrix_size; i++)
            mpi_matrix = new float[N];
        
        // 从第rank_0节点接收初始数据
    	for(int i=0; i<matrix_size; i++)
        {
            MPI_Recv(mpi_matrix[i], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }


        for(int i=0; i<matrix_size; i++)
        {
            // elimination
            if(0 == i){
                for(int k=0; k<rank; k++){
                    // 从src节点接收第k行division操作后的结果
                    int src = k%m;
                    MPI_Recv(row_k, N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    // 对自己负责的行做消去
                    matrix_elimination(mpi_matrix, matrix_size, row_k, N, k, i);
                }
            }
            else{
                for(int j=0; j<m-1;j++){
                    int k = _n *i + rank - (m-1 -j);
                MPI_Recv(row_k, N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                // 对自己负责的行做消去
                matrix_elimination(mpi_matrix, matrix_size, row_k, N, k, i);
                }
            }

            // division
            int k = _n *i + rank;

        }

        // // 此节点中最后行对应原始矩阵中的行数
        // int end_row = (matrix_size - 1) * _n + rank; 

        // for(int k=0; k<=end_row; k++)
        // {
        //     src = k%m;  // 计算外层循环到第k行时，第k行的division操作是由哪个节点负责
        //     if(rank == src)  // 由本节点负责
        //     {
        //         // division
        //         int i = 
        //     }
        // }


        

}


void elimination(float **matrix/*矩阵*/, int n_row/*行数*/, float *row_k/*第k行*/, 
                int dimension/*每行维度*/, int k/*外层循环进行到k*/)
{
    for(int i=0; i<n_row; i++)
    {
        for(int j=k+1; j<dimension; j++)
            matrix[i][j] = matrix[i][j] - matrix[i][k]*row_k[j];
        matrix[i][k] = 0.0;
    }
}

void matrix_elimination(float **matrix, int n_row, float *row_k, 
                        int dimension, int k, int row_No/*row_No之前的不进行操作*/)
{
    for(int i=row_No; i<n_row; i++)
    {
        for(int j=k+1; j<dimension; j++)
            matrix[i][j] = matrix[i][j] - matrix[i][k]*row_k[j];
        matrix[i][k] = 0.0;
    }
}


