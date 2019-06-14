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
 
    if(rank < q && q != 0)
        matrix_size = _n+1;
    else
        matrix_size = _n;
        
    // 用来存这一节点所负责的行
    float **mpi_matrix = new float*[matrix_size];
    for(int i=0; i<matrix_size; i++)
        mpi_matrix[i] = new float[N];
    
    float *row_k = new float[N];
    
    //cout << proc_name << ':' << "rank[" << rank << "]: size = " << matrix_size << endl;

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

        int count = 0;
        // 将各行的值发送给各线程
        for(int i = 0; i<N; i++)
        {
            // 计算第i行应该要发到的节点是哪个
            int dest = i % m;
            if(dest != rank)
                MPI_Send(A[i], N/*size*/, MPI_FLOAT/*type*/, dest/*dest*/, 0, MPI_COMM_WORLD);
            else{
                // 自己负责的行移动至mpi_matrix里面
                for(int j = 0; j<N; j++)
                    mpi_matrix[count][j] = A[i][j];
                count ++;
            }
        }
        // cout << proc_name << ':' << "rank[" << rank << "]: Sending all the data "  << endl;
        // cout << proc_name << ':' << "rank[" << rank << "]: mpi_matrix " << endl;
        // show_matrix(mpi_matrix, matrix_size, N);

        for(int i=0; i<matrix_size; i++)
        {
            //cout << proc_name << ':' << "rank[" << rank << "]: 进行到第 " << i << " 行" << endl;
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
                for(int j=1; j<=m-1;j++){
                    int k = m *(i-1) + j + rank;
                    int src = k%m;
                    MPI_Recv(row_k, N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    // 对自己负责的行做消去
                    matrix_elimination(mpi_matrix, matrix_size, row_k, N, k, i);
                }
            }

            // division
            int k = m *i + rank;
            for(int j=k+1; j<N; j++)
                mpi_matrix[i][j] = mpi_matrix[i][j] / mpi_matrix[i][k];
            mpi_matrix[i][k] = 1.0;
            // elimination
            matrix_elimination(mpi_matrix, matrix_size, mpi_matrix[i], N, k, i+1);

            // 把division之后的结果发给其他各节点
            int end_row_rank = (N-1) % m;  // 原矩阵最后一行对应的节点
            int end_row = N-1;
            int real_row = i*m + rank;
            // 看其对应的原矩阵的行是不是最后m-1行，如果是则最后发送的目标节点并不是全部节点
            if((i == matrix_size-1) && (end_row-m+2 <=real_row) &&(real_row <= end_row))
            {
                for(int row=real_row+1; row<=end_row; row++){
                	int dest = row % m;
                    if(dest != rank)
                        MPI_Send(mpi_matrix[i], N, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
                }
            }
            else
            {
                for(int dest=0; dest<m; dest++)
                    if(dest != rank) 
                        MPI_Send(mpi_matrix[i], N, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
            }

        }
        
        count = 0;
        for(int i = 0; i<N; i++)
        {
            int src = i % m;
            if(rank != src){
                //cout << proc_name << ':' << "rank[" << rank << "]: Receiving " << i << " row" << endl;
                MPI_Recv(A[i], N/*size*/, MPI_FLOAT/*type*/, src/*dest*/, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            else{
                for(int j = 0; j<N; j++)
                    A[i][j] =  mpi_matrix[count][j];
                count ++;
            }
        }
        //cout << proc_name << ':' << "rank[" << rank << "]: Done "  << endl;
        
        
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
        // 从第rank_0节点接收初始数据
    	for(int i=0; i<matrix_size; i++)
        {
            MPI_Recv(mpi_matrix[i], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }


        for(int i=0; i<matrix_size; i++)
        {
            //cout << proc_name << ':' << "rank[" << rank << "]: 进行到第 " << i << " 行" << endl;

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
                for(int j=1; j<=m-1;j++){
                    int k = m *(i-1) + j + rank;
                    int src = k%m;
                    MPI_Recv(row_k, N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    // 对自己负责的行做消去
                    matrix_elimination(mpi_matrix, matrix_size, row_k, N, k, i);
                }
            }

            // division
            int k = m *i + rank;
            for(int j=k+1; j<N; j++)
                mpi_matrix[i][j] = mpi_matrix[i][j] / mpi_matrix[i][k];
            mpi_matrix[i][k] = 1.0;
            // elimination
            matrix_elimination(mpi_matrix, matrix_size, mpi_matrix[i], N, k, i+1);


            //cout << proc_name << ':' << "rank[" << rank << "]: Sending.... " << endl;
            // 把division之后的结果发给其他各节点
            int end_row_rank = (N-1) % m;  // 原矩阵最后一行对应的节点
            int end_row = N-1;
            int real_row = i*m + rank;
            // 看其对应的原矩阵的行是不是最后m-1行，如果是则最后发送的目标节点并不是全部节点
            if((i == matrix_size-1) && (end_row-m+2 <=real_row) &&(real_row <= end_row))
            {
                for(int row=real_row+1; row<=end_row; row++){
                	int dest = row % m;
                    if(dest!=rank)
                        MPI_Send(mpi_matrix[i], N, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
                }
            }
            else
            {
                for(int dest=0; dest<m; dest++)
                    if(dest != rank) 
                        MPI_Send(mpi_matrix[i], N, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
            }

        }
        
        // cout << proc_name << ':' << "rank[" << rank << "]: Sending final data " << endl;
        // show_matrix(mpi_matrix, matrix_size, N);
        // 发送全部数据到rank_0节点
        for(int i = 0; i<matrix_size; i++)
        {
            MPI_Send(mpi_matrix[i], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        }
    }

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


