#include <iostream>
#include <windows.h>
#include <nmmintrin.h>
#include <immintrin.h>
#include <mpi.h>
#include <omp.h>
using namespace std;
const int N = 1000;
const int thread_count = 4;
float A[N][N];
float matrix[N][N];
void init()
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            matrix[i][j] = 0;
        }
        matrix[i][i] = 1.0;
        for (int j = i + 1; j < N; j++)
            matrix[i][j] = rand();
    }
    for (int k = 0; k < N; k++)
    {
        for (int i = k + 1; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                matrix[i][j] += matrix[k][j];
            }
        }
    }
}
void reset()
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i][j] = matrix[i][j];
        }
    }
}
void print()
{
    for (int i = 0; i < 10; i++)
        cout << A[0][i] << " ";
    cout << endl;
}
void normal()
{
    for (int k = 0; k < N; k++)
    {
        for (int i = k + 1; i < N; i++)
            A[k][i] = A[k][i] / A[k][k];
        A[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            for (int j = k + 1; j < N; j++)
                A[i][j] -= A[i][k] * A[k][j];
            A[i][k] = 0;
        }
    }
}
void LU_ROW(int rank, int size)
{
    int block = N / size, remain = N % size;
    int r1 = rank * block;
    int r2 = r1 + block;
    if (rank == size - 1)
        r2 += remain;
    for (int k = 0; k < N; k++)
    {
        if (k >= r1 && k < r2)
        {
            for (int j = k + 1; j < N; j++)
                A[k][j] = A[k][j] / A[k][k];
            A[k][k] = 1.0;
            for (int j = 0; j < size; j++)
                if (j != rank)
                    MPI_Send(&A[k], N, MPI_FLOAT, j, 2, MPI_COMM_WORLD);
        }
        else
            MPI_Recv(&A[k], N, MPI_FLOAT, k / block, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = r1; i < r2 && i < N; i++)
        {
            if (i >= k + 1)
            {
                for (int j = k + 1; j < N; j++)
                    A[i][j] = A[i][j] - A[i][k] * A[k][j];
                A[i][k] = 0.0;
            }
        }
    }
}
void Main_ROW()
{
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int block = N / size, remain = N % size;
    if (rank == 0)
    {
        for (int i = 1; i < size; i++)
        {
            if (i != size - 1)
                for (int j = 0; j < block; j++)
                    MPI_Send(&A[i * block + j], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            else
                for (int j = 0; j < block + remain; j++)
                    MPI_Send(&A[i * block + j], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
        }
        LU_ROW(rank, size);
        for (int i = 1; i < size; i++)
        {
            if (i != size - 1)
                for (int j = 0; j < block; j++)
                    MPI_Recv(&A[i * block + j], N, MPI_FLOAT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            else
                for (int j = 0; j < block + remain; j++)
                    MPI_Recv(&A[i * block + j], N, MPI_FLOAT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    else
    {
        if (rank != size - 1)
            for (int j = 0; j < block; j++)
                MPI_Recv(&A[rank * block + j], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        else
            for (int j = 0; j < block + remain; j++)
                MPI_Recv(&A[rank * block + j], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        LU_ROW(rank, size);
        if (rank != size - 1)
            for (int j = 0; j < block; j++)
                MPI_Send(&A[rank * block + j], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        else
            for (int j = 0; j < block + remain; j++)
                MPI_Send(&A[rank * block + j], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
    }
}
void LU_COL(int rank, int size)
{
    for (int k = 0; k < N; k++)
    {
        int tmp = k % size;
        if (tmp == rank)
        {
            for (int j = k + 1; j < N; j++)
                A[k][j] = A[k][j] / A[k][k];
            A[k][k] = 1.0;
            for (int j = 0; j < size; j++)
                if (j != rank)
                    MPI_Send(&A[k], N, MPI_FLOAT, j, 2, MPI_COMM_WORLD);
        }
        else
            MPI_Recv(&A[k], N, MPI_FLOAT, tmp, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = k + 1; i < N; i++)
        {
            if (i % size == rank)
            {
                for (int j = k + 1; j < N; j++)
                    A[i][j] = A[i][j] - A[i][k] * A[k][j];
                A[i][k] = 0.0;
            }
        }
    }
}
void Main_COL()
{
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        for (int i = 0; i < N; i++)
        {
            int tmp = i % size;
            if (tmp != rank)
                MPI_Send(&A[i], N, MPI_FLOAT, tmp, 0, MPI_COMM_WORLD);
        }
        LU_COL(rank, size);
        for (int i = 0; i < N; i++)
        {
            int tmp = i % size;
            if (tmp != rank)
                MPI_Recv(&A[i], N, MPI_FLOAT, tmp, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    else
    {
        for (int i = rank; i < N; i += size)
            MPI_Recv(&A[i], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        LU_COL(rank, size);
        for (int i = rank; i < N; i += size)
            MPI_Send(&A[i], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
    }
}
int main()
{
    int epoch = 10;
    long long begin, end, freq;
    double timer;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    init();
    timer = 0;
    for (int i = 0; i < epoch; i++)
    {
        reset();
        QueryPerformanceCounter((LARGE_INTEGER*)&begin);
        normal();
        QueryPerformanceCounter((LARGE_INTEGER*)&end);
        timer += (end - begin) * 1000.0 / freq;
    }
    print();
    cout << "normal:  " << timer / epoch << "ms" << endl;
    MPI_Init(NULL, NULL);
    timer = 0;
    for (int i = 0; i < epoch; i++)
    {
        reset();
        QueryPerformanceCounter((LARGE_INTEGER*)&begin);
        Main_ROW();
        QueryPerformanceCounter((LARGE_INTEGER*)&end);
        timer += (end - begin) * 1000.0 / freq;
    }
    print();
    cout << "Main_ROW:  " << timer / epoch << "ms" << endl;
    timer = 0;
    for (int i = 0; i < epoch; i++)
    {
        reset();
        QueryPerformanceCounter((LARGE_INTEGER*)&begin);
        Main_COL();
        QueryPerformanceCounter((LARGE_INTEGER*)&end);
        timer += (end - begin) * 1000.0 / freq;
    }
    print();
    cout << "Main_COL:  " << timer / epoch << "ms" << endl;
    MPI_Finalize();
    return 0;
}
