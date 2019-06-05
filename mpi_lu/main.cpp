#include <mpi.h>
#include "matrix.h"
#include <stdio.h>
#include <sys/time.h>
#include <iostream>

using namespace std;
const int N = 300;

void thread_perform(int rank);

int main(int argc, char ** argv)
{
    int rank;
	int n_proc;  // the number of running threads
	

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    thread_perform(rank);
    
    MPI_Finalize();
    return 0;
}

void thread_perform(int rank)
{
	if(0 == rank)
	{
	
    	struct timeval start, end;
        // 因为矩阵中一行和线程是一一对应的关系，所以线程数得等于行数
        int size = MPI::COMM_WORLD.Get_size();
        if(size != N)
        {
            cout << "开启的线程数和矩阵规模不一致，请重新设置" << endl;
            MPI_Finalize();
            return;
        }

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

		//show_matrix(A, N);
        
        gettimeofday(&start, NULL);

        // 将各行的值发送给各线程
        for(int i = 1; i<N; i++)
        {
            MPI_Send(A[i], N/*size*/, MPI_FLOAT/*type*/, i/*dest*/, 0, MPI_COMM_WORLD);
        }

        float *row_i = new float[N];
        for(int i=0; i<N; i++)
            row_i[i] = A[0][i];
        
        // division part
        for(int j=1; j<N; j++)
            A[0][j] = A[0][j]/A[0][0];
        A[0][0] = 1.0;
        
        for(int i=1; i<N; i++)
        {
            MPI_Send(A[0], N/*size*/, MPI_FLOAT/*type*/, i/*dest*/, 0, MPI_COMM_WORLD);
        }
        for(int i=1; i<N; i++)
            MPI_Recv(A[i], N, MPI_FLOAT, i/*source*/, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        gettimeofday(&end, NULL);
        unsigned long time_interval = 1000000*(end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
        //show_matrix(A, N);

        //cout << "standard lu: "<< endl;
        //show_matrix(A_copy, N);
        cout << "time is " << time_interval << endl;
        
        if(!ls_same(A, A_copy, N))
        	cout << "problem" << endl;
        else
        	cout << "correct" << endl;
        
	}
    else
    {
        float *row_i = new float[N];  // 一行的数据
        float *row_k = new float[N];  // 用来保存k行做完除法后的数据
        MPI_Recv(row_i, N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);  // 接收数据
        //cout << "rank" << rank << "has received the data" << endl;
        // show_vector(row_i, N);
        // 接收完数据后，当k行做完除法后，接收k行的数据，做消去运算
        for(int k = 0; k<rank; k++)
        {
            MPI_Recv(row_k, N, MPI_FLOAT, k/*source*/, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for(int j = k+1; j<N; j++)
                row_i[j] = row_i[j] - row_i[k]*row_k[j];
            row_i[k] = 0.0;
        }
        // division part
        for(int j=rank+1; j<N; j++)
            row_i[j] = row_i[j]/row_i[rank];
        row_i[rank] = 1.0;

        for(int i=rank+1; i<N; i++)
        {
            MPI_Send(row_i, N/*size*/, MPI_FLOAT/*type*/, i/*dest*/, 0, MPI_COMM_WORLD);
        }
        
        MPI_Send(row_i, N/*size*/, MPI_FLOAT/*type*/, 0/*dest*/, 0, MPI_COMM_WORLD);
    }
}


