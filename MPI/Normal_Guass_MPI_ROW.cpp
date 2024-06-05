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
void LU_COL_OpenMP(int rank, int size)
{
#pragma omp parallel num_threads(thread_count)
    for (int k = 0; k < N; k++)
    {
        int tmp = k % size;
        if (tmp == rank)
        {
            __m128 t1, t2, t3;
            float tmp[4] = { A[k][k], A[k][k], A[k][k], A[k][k] };
            t1 = _mm_loadu_ps(tmp);
#pragma omp for schedule(static)
            int j;
            for (j = k + 1; j + 4 < N; j += 4)
            {
                t2 = _mm_loadu_ps(A[k] + j);
                t3 = _mm_div_ps(t2, t1);
                _mm_storeu_ps(A[k] + j, t3);
            }
            for (; j < N; j++)
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
                __m128 t1, t2, t3;
                float tmp[4] = { A[k][k], A[k][k], A[k][k], A[k][k] };
                t1 = _mm_loadu_ps(tmp);
#pragma omp for schedule(static)
                int j;
                for (j = k + 1; j + 4 < N; j += 4)
                {
                    t2 = _mm_loadu_ps(A[i] + j);
                    t3 = _mm_loadu_ps(A[k] + j);
                    t3 = _mm_mul_ps(t1, t3);
                    t2 = _mm_sub_ps(t2, t3);
                    _mm_storeu_ps(A[i] + j, t2);
                }
                for (; j < N; j++)
                    A[i][j] = A[i][j] - A[i][k] * A[k][j];
                A[i][k] = 0.0;
            }
        }
    }
}
void Main_COL_OpenMP()
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
        LU_COL_OpenMP(rank, size);
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
        LU_COL_OpenMP(rank, size);
        for (int i = rank; i < N; i += size)
            MPI_Send(&A[i], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
    }
}
int main()
{
    int epoch = 1;
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
        Main_COL();
        QueryPerformanceCounter((LARGE_INTEGER*)&end);
        timer += (end - begin) * 1000.0 / freq;
    }
    print();
    cout << "Main_COL:  " << timer / epoch << "ms" << endl;
    timer = 0;
    for (int i = 0; i < epoch; i++)
    {
        reset();
        QueryPerformanceCounter((LARGE_INTEGER*)&begin);
        Main_COL_OpenMP();
        QueryPerformanceCounter((LARGE_INTEGER*)&end);
        timer += (end - begin) * 1000.0 / freq;
    }
    print();
    cout << "Main_COL_OpenMP:  " << timer / epoch << "ms" << endl;
    MPI_Finalize();
    return 0;
}
