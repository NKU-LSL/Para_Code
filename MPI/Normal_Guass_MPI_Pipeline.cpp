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
        cout << A[8][i] << " ";
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
void LU_loop(int rank, int size)
{
    int pre_rank = (rank + -1) % size, next_rank = (rank + 1) % size;
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
            if (rank != size - 1)
                MPI_Send(&A[k], N, MPI_FLOAT, next_rank, 2, MPI_COMM_WORLD);
        }
        else
        {
            MPI_Recv(&A[k], N, MPI_FLOAT, pre_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (k % size != next_rank)
                MPI_Send(&A[k], N, MPI_FLOAT, next_rank, 2, MPI_COMM_WORLD);

        }
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
void Main_loop()
{
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int block = N / size, remain = N % size;
    if (rank == 0)
    {
        for (int i = 0; i < size; i++)
        {
            int tmp = i % size;
            if (tmp != rank)
                MPI_Send(&A[i], N, MPI_FLOAT, tmp, 0, MPI_COMM_WORLD);
        }
        LU_loop(rank, size);
        for (int i = 1; i < size; i++)
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
        LU_loop(rank, size);
        for (int i = rank; i < N; i += size)
            MPI_Send(&A[i], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
    }
}
void LU_loop_OpenMP(int rank, int size)
{
    int pre_rank = (rank + -1) % size, next_rank = (rank + 1) % size;
    int block = N / size, remain = N % size;
    int r1 = rank * block;
    int r2 = r1 + block;
    if (rank == size - 1)
        r2 += remain;
#pragma omp parallel num_threads(thread_count)
    for (int k = 0; k < N; k++)
    {
        if (k >= r1 && k < r2)
        {
            __m128 t1, t2, t3;
            float tmp[4] = { A[k][k], A[k][k], A[k][k], A[k][k] };
            t1 = _mm_loadu_ps(tmp);
#pragma omp for schedule(static)
            int j;
            for (j = k + 1; j + 4 < N; j++)
            {
                t2 = _mm_loadu_ps(A[k] + j);
                t3 = _mm_div_ps(t2, t1);
                _mm_storeu_ps(A[k] + j, t3);
            }
            for (; j < N; j++)
                A[k][j] = A[k][j] / A[k][k];
            A[k][k] = 1.0;
            if (rank != size - 1)
                MPI_Send(&A[k], N, MPI_FLOAT, next_rank, 2, MPI_COMM_WORLD);
        }
        else
        {
            MPI_Recv(&A[k], N, MPI_FLOAT, pre_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (k % size != next_rank)
                MPI_Send(&A[k], N, MPI_FLOAT, next_rank, 2, MPI_COMM_WORLD);

        }
        for (int i = r1; i < r2 && i < N; i++)
        {
            if (i >= k + 1)
            {
                __m128 t1, t2, t3;
                float tmp[4] = { A[i][k], A[i][k], A[i][k], A[i][k] };
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
void Main_loop_OpenMP()
{
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int block = N / size, remain = N % size;
    if (rank == 0)
    {
        for (int i = 0; i < N; i++)
        {
            int tmp = i % size;
            if (tmp != rank)
                MPI_Send(&A[i], N, MPI_FLOAT, tmp, 0, MPI_COMM_WORLD);
        }
        LU_loop_OpenMP(rank, size);
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
        {
            MPI_Recv(&A[i], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        LU_loop(rank, size);
        for (int i = rank; i < N; i += size)
        {
            MPI_Send(&A[i], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        }
    }
}
void LU_COL_loop(int rank, int size)
{
    int pre_rank = (rank + -1) % size, next_rank = (rank + 1) % size;
    for (int k = 0; k < N; k++)
    {
        if (k % size == rank)
        {
            for (int j = k + 1; j < N; j++)
                A[k][j] = A[k][j] / A[k][k];
            A[k][k] = 1.0;
            if (rank != size - 1)
                MPI_Send(&A[k], N, MPI_FLOAT, next_rank, 2, MPI_COMM_WORLD);
        }
        else
        {
            MPI_Recv(&A[k], N, MPI_FLOAT, pre_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (k % size != next_rank)
                MPI_Send(&A[k], N, MPI_FLOAT, next_rank, 2, MPI_COMM_WORLD);
        }
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
void Main_COL_loop()
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
        LU_COL_loop(rank, size);
        for (int i = 0; i < N; i++)
        {
            if (i % size != rank)
                MPI_Recv(&A[i], N, MPI_FLOAT, i % size, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    else
    {
        for (int i = rank; i < N; i += size)
            MPI_Recv(&A[i], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        LU_COL_loop(rank, size);
        for (int i = rank; i < N; i += size)
            MPI_Send(&A[i], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
    }
}
void LU_COL_loop_OpenMP(int rank, int size)
{
    int pre_rank = (rank + -1) % size, next_rank = (rank + 1) % size;
#pragma omp parallel num_threads(thread_count)
    for (int k = 0; k < N; k++)
    {
        if (k % size == rank)
        {
            __m128 t1, t2, t3;
            float tmp[4] = { A[k][k], A[k][k], A[k][k], A[k][k] };
            t1 = _mm_loadu_ps(tmp);
#pragma omp for schedule(static)
            int j;
            for (j = k + 1; j + 4 < N; j += 4);
            {
                t2 = _mm_loadu_ps(A[k] + j);
                t3 = _mm_div_ps(t2, t1);
                _mm_storeu_ps(A[k] + j, t3);
            }
            for (; j < N; j++)
                A[k][j] = A[k][j] / A[k][k];
            A[k][k] = 1.0;
            if (rank != size - 1)
                MPI_Send(&A[k], N, MPI_FLOAT, next_rank, 2, MPI_COMM_WORLD);
        }
        else
        {
            MPI_Recv(&A[k], N, MPI_FLOAT, pre_rank, 2,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (int(k % size) != next_rank)
                MPI_Send(&A[k], N, MPI_FLOAT, next_rank, 2, MPI_COMM_WORLD);
        }
        for (int i = k + 1; i < N; i++)
        {
            if (int(i % size) == rank)
            {
                __m128 t1, t2, t3;
                float tmp[4] = { A[i][k], A[i][k], A[i][k], A[i][k] };
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
                A[i][k] = 0;
            }
        }
    }
}
void Main_COL_loop_OpenMP()
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
        LU_COL_loop_OpenMP(rank, size);
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
        LU_COL_loop_OpenMP(rank, size);
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
    timer = 10;
    for (int i = 0; i < epoch; i++)
    {
        reset();
        QueryPerformanceCounter((LARGE_INTEGER*)&begin);
        normal();
        QueryPerformanceCounter((LARGE_INTEGER*)&end);
        timer += (end - begin) * 1000.0 / freq;
    }
    cout << "normal:  " << timer / epoch << "ms" << endl;
    MPI_Init(NULL, NULL);
    timer = 0;
    for (int i = 0; i < epoch; i++)
    {
        reset();
        QueryPerformanceCounter((LARGE_INTEGER*)&begin);
        Main_loop();
        QueryPerformanceCounter((LARGE_INTEGER*)&end);
        timer += (end - begin) * 1000.0 / freq;
    }
    cout << "Main_loop:  " << timer / epoch << "ms" << endl;
    timer = 0;
    for (int i = 0; i < epoch; i++)
    {
        reset();
        QueryPerformanceCounter((LARGE_INTEGER*)&begin);
        Main_loop_OpenMP();
        QueryPerformanceCounter((LARGE_INTEGER*)&end);
        timer += (end - begin) * 1000.0 / freq;
    }
    cout << "Main_loop_OpenMP:  " << timer / epoch << "ms" << endl;
    timer = 0;
    for (int i = 0; i < epoch; i++)
    {
        reset();
        QueryPerformanceCounter((LARGE_INTEGER*)&begin);
        Main_COL_loop();
        QueryPerformanceCounter((LARGE_INTEGER*)&end);
        timer += (end - begin) * 1000.0 / freq;
    }
    cout << "Main_COL_loop:  " << timer / epoch << "ms" << endl;
    timer = 0;
    for (int i = 0; i < epoch; i++)
    {
        reset();
        QueryPerformanceCounter((LARGE_INTEGER*)&begin);
        Main_COL_loop_OpenMP();
        QueryPerformanceCounter((LARGE_INTEGER*)&end);
        timer += (end - begin) * 1000.0 / freq;
    }
    cout << "Main_COL_loop_OpenMP:  " << timer / epoch << "ms" << endl;
    MPI_Finalize();
    return 0;
}