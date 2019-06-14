/*
基于块划分的流水线LU高斯消去算法
每一个节点分配N/m（这里的表示是不精确的）行的数据
设：n_ = (N-N%m)/m，对于rank号节点，其分配的行的范围为：
    rank != m-1时，[rank*_n, rank*_n+_n-1]
    rank == m-1时，[rank*_n, N-1]
当一个节点当中的一行进行完了除法操作以后
此节点将除法操作得到的结果发送给下一个节点
而不是发送给下面的每一个节点
同样，当一个节点收到上面节点发送过来的division操作的结果时
其对自己所负责的行进行消去操作，之后又将刚刚收到的division操作的结果发给下一节点
*/

#include <omp.h>
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
	if(0 == rank)
	{
	
    	struct timeval start, end;
        // 因为矩阵中一行和线程是一一对应的关系，所以线程数得等于行数
        int m = MPI::COMM_WORLD.Get_size();  // m是总的mpi节点个数
        int _n = (N - N%m) / m;

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

            // // 将division结果发给后面的节点
            // for(int dest=rank+1; dest<m; dest++){
            //     MPI_Send(A[k], N, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
            // }

            // 将division结果发给下一个节点
            int dest = rank + 1;
            if(dest <= m-1)
            	MPI_Send(A[k], N, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);

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
        long long time_interval = 1000000*(end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
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
    	
        // 首先计算这一节点所负责的行的范围
        int m = MPI::COMM_WORLD.Get_size();  // m是总的mpi节点个数
        int _n = (N - N%m) / m;
        int begin_row, end_row;  // 该节点所负责的起始行号和结束行号
        begin_row = rank * _n;
        if(rank == m-1)
            end_row = N-1; 
        else
            end_row = rank * _n + _n-1;
        
        // 创建空间存放要负责行
        int matrix_size = end_row - begin_row + 1;
        float **mpi_matrix = new float*[matrix_size];
        for(int i=0; i<matrix_size; i++)
            mpi_matrix[i] = new float[N];

        // 从0号mpi节点接受这几行的数据
        for(int i=0; i<matrix_size; i++)
        {
            MPI_Recv(mpi_matrix[i]/*addr*/, N/*count*/, MPI_FLOAT/*type*/, 0/*sour*/, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        //cout << proc_name << ':' << "rank[" << rank << "]: recveid all the data" << endl;

        // 接收begin_row次，对自己负责的这些行做消去操作
        float *row_k = new float[N];  // 用来存收到的第k行做division后的结果
        for(int k=0; k<begin_row; k++)
        {
            // 从上一个节点接收
            int src = rank - 1;
            // 接收第k行做division后的结果
            MPI_Recv(row_k, N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // 对所负责的行做消去操作
            matrix_elimination(mpi_matrix, matrix_size, row_k, N, k, 0);
            //cout << proc_name << ':' << "rank[" << rank << "]: elimination k = " << k << endl;
            // 做完消去操作后要把k行的division结果发送给下一个节点
            int dest = rank+1;
            if(dest <= m-1){
                MPI_Send(row_k, N, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
            }
        }

        
        // 对自己负责的行做division操作，发送结果，并对自己负责的行做消去
        for(int k=begin_row; k <= end_row; k++)
        {
            // division
            int i = k - begin_row;
            //cout << "i = " << i << endl;
            for(int j=k+1; j<N; j++)
                mpi_matrix[i][j] = mpi_matrix[i][j] / mpi_matrix[i][k];
            mpi_matrix[i][k] = 1.0;

            //cout << proc_name << ':' << "rank[" << rank << "]: division k = " << k << endl;

            // 将division结果发给后面的节点
            if(rank != m-1){
                // for(int dest=rank+1; dest<m; dest++){
                //     MPI_Send(mpi_matrix[i], N, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
                // }
                int dest = rank + 1;
                if(dest <= m-1)
                    MPI_Send(mpi_matrix[i], N, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
            }

            // 对剩下的行做消去
            int rows_left = end_row - k;
            if(rows_left <= 0)
                break;
            
            matrix_elimination(mpi_matrix, matrix_size, mpi_matrix[i], N, k, i+1);
            
        }

        //cout << proc_name << ':' << "rank[" << rank << "]: " << endl;
        //show_matrix(mpi_matrix, matrix_size, N);

        // 计算全部完成后把自己负责的行发回0号节点
        // 这里采用阻塞式发送
        for(int i=0; i<matrix_size; i++){
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




void matrix_elimination(float **matrix, int n_row, float *row_k, int dimension, int k, int row_No/**/)
{
    #pragma omp parallel for num_threads(2)
    for(int i=row_No; i<n_row; i++)
    {
        for(int j=k+1; j<dimension; j++)
            matrix[i][j] = matrix[i][j] - matrix[i][k]*row_k[j];
        matrix[i][k] = 0.0;
    }
}

