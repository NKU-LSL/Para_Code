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
void print()
{
    for (int i = 0; i < 10; i++)
        cout << A[0][i] << " ";
    cout << endl;
}
void LU(int rank, int size)
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
void Main()
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
        LU(rank, size);
        for (int i = 1; i < size; i++)
        {
            if (i != size - 1)
                for (int j = 0; j < block; j++)
                    MPI_Recv(&A[i * block + j], N, MPI_FLOAT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            else 
                for (int j = 0; j < block + remain; j++)
                    MPI_Recv(&A[i* block + j], N, MPI_FLOAT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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
        LU(rank, size);
        if (rank != size - 1)
            for (int j = 0; j < block; j++)
                MPI_Send(&A[rank * block + j], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        else
            for (int j = 0; j < block + remain; j++)
                MPI_Send(&A[rank * block + j], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
    }
}
void LU2(int rank, int size)
{
    int block = N / size;
    int remain = N % size;
    int start = rank * block + (rank < remain ? rank : remain);
    int end = start + block + (rank < remain ? 1 : 0);

    for (int k = 0; k < N; k++)
    {
        if (k >= start && k < end)
        {
            for (int j = k + 1; j < N; j++)
                A[k][j] = A[k][j] / A[k][k];
            A[k][k] = 1.0;
            for (int j = 0; j < size; j++)
                if (j != rank)
                    MPI_Send(&A[k], N, MPI_FLOAT, j, 2, MPI_COMM_WORLD);
        }
        else
            MPI_Recv(&A[k], N, MPI_FLOAT, MPI_ANY_SOURCE, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int i = start; i < end && i < N; i++)
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

void Main2()
{
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int block = N / size;
    int remain = N % size;

    if (rank == 0)
    {
        for (int i = 1; i < size; i++)
        {
            int start = i * block + (i < remain ? i : remain);
            int end = start + block + (i < remain ? 1 : 0);
            for (int j = start; j < end; j++)
                MPI_Send(&A[j], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
        }
        LU2(rank, size);
        for (int i = 1; i < size; i++)
        {
            int start = i * block + (i < remain ? i : remain);
            int end = start + block + (i < remain ? 1 : 0);
            for (int j = start; j < end; j++)
                MPI_Recv(&A[j], N, MPI_FLOAT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    else
    {
        int start = rank * block + (rank < remain ? rank : remain);
        int end = start + block + (rank < remain ? 1 : 0);
        for (int j = start; j < end; j++)
            MPI_Recv(&A[j], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        LU2(rank, size);

        for (int j = start; j < end; j++)
            MPI_Send(&A[j], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
    }
}

void LU_avg(int rank, int size)
{
    int block = N / size, remain = N % size;
    int r1 = rank * block;
    int r2 = r1 + block;
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
        else if (k == size * block + rank && rank < remain)
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
void Main_avg()
{
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int block = N / size, remain = N % size;
    if (rank == 0)
    {
        for (int i = 1; i < size; i++)
        {
            for (int j = 0; j < block; j++)
                MPI_Send(&A[i * block + j], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            if (rank < remain)
                MPI_Send(&A[size * block + rank], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
        }
        LU_avg(rank, size);
        for (int i = 1; i < size; i++)
        {
            for (int j = 0; j < block; j++)
                MPI_Recv(&A[i * block + j], N, MPI_FLOAT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (rank < remain)
                MPI_Recv(&A[size * block + rank], N, MPI_FLOAT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    else
    {
        for (int j = 0; j < block; j++)
            MPI_Recv(&A[rank * block + j], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (rank < remain)
            MPI_Recv(&A[size * block + rank], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        LU_avg(rank, size);
        for (int j = 0; j < block; j++)
            MPI_Send(&A[rank * block + j], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        if (rank < remain)
            MPI_Send(&A[size * block + rank], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
    }
}
void LU_OpenMP(int rank, int size)
{
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
            for (j = k + 1; j + 4 < N; j += 4)
            {
                t2 = _mm_loadu_ps(A[k] + j);
                t3 = _mm_div_ps(t2, t1);
                _mm_storeu_ps(A[k] + j, t3);
            }
            for (; j < N; j++)
                A[k][j] = A[k][j] / A[k][k];
            A[k][k] = 1.0;
            for (int h = 0; h < size; h++)
                if (h != rank)
                    MPI_Send(&A[k], N, MPI_FLOAT, h, 2, MPI_COMM_WORLD);
        }
        else
            MPI_Recv(&A[k], N, MPI_FLOAT, k / block, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = r1; i < r2 && i < N; i++)
        {
            if (i >= k + 1)
            {
                __m128 t1, t2, t3;
                float temp2[4] = { A[i][k], A[i][k], A[i][k], A[i][k] };
                t1 = _mm_loadu_ps(temp2);
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
void Main_OpenMP()
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
                    MPI_Send(&A[(size - 1) * block + j], N, MPI_FLOAT, size - 1, 0, MPI_COMM_WORLD);
        }
        LU_OpenMP(rank, size);
        for (int i = 1; i < size; i++)
        {
            if (i != size - 1)
                for (int j = 0; j < block; j++)
                    MPI_Recv(&A[i * block + j], N, MPI_FLOAT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            else 
                for (int j = 0; j < block + remain; j++)
                    MPI_Recv(&A[(size - 1) * block + j], N, MPI_FLOAT, size - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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
        LU_OpenMP(rank, size);
        if (rank != size - 1)
            for (int j = 0; j < block; j++)
                MPI_Send(&A[rank * block + j], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        else
            for (int j = 0; j < block + remain; j++)
                MPI_Send(&A[rank * block + j], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
    }
}
int main()
{
    MPI_Init(NULL, NULL);
    int rank, size;

    // 获取进程的秩和总的进程数
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
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
    timer = 0;
    for (int i = 0; i < epoch; i++)
    {
        reset();
        QueryPerformanceCounter((LARGE_INTEGER*)&begin);
        Main();
        QueryPerformanceCounter((LARGE_INTEGER*)&end);
        timer += (end - begin) * 1000.0 / freq;
    }
    print();
    cout << "Main:  " << timer / epoch << "ms" << endl;
    timer = 0;
    for (int i = 0; i < epoch; i++)
    {
        reset();
        QueryPerformanceCounter((LARGE_INTEGER*)&begin);
        Main2();
        QueryPerformanceCounter((LARGE_INTEGER*)&end);
        timer += (end - begin) * 1000.0 / freq;
    }
    print();
    cout << "Main_avg2:  " << timer / epoch << "ms" << endl;
    timer = 0;
    for (int i = 0; i < epoch; i++)
    {
        reset();
        QueryPerformanceCounter((LARGE_INTEGER*)&begin);
        Main_avg();
        QueryPerformanceCounter((LARGE_INTEGER*)&end);
        timer += (end - begin) * 1000.0 / freq;
    }
    print();
    cout << "Main_avg:  " << timer / epoch << "ms" << endl;
    timer = 0;
    for (int i = 0; i < epoch; i++)
    {
        reset();
        QueryPerformanceCounter((LARGE_INTEGER*)&begin);
        Main_OpenMP();
        QueryPerformanceCounter((LARGE_INTEGER*)&end);
        timer += (end - begin) * 1000.0 / freq;
    }
    print();
    cout << "Main_OpenMP:  " << timer / epoch << "ms" << endl;
    MPI_Finalize();
    return 0;
}
