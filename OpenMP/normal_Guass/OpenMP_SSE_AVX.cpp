#include<iostream>
#include<windows.h>
#include<nmmintrin.h>
#include<immintrin.h>
#include<omp.h>
using namespace std;
const int N =1000;
float A[N][N];
float matrix[N][N];
const int NUM_THREADS = 7;
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
void Openmp_SSE_static()
{
    int i, j, k;
    #pragma omp parallel num_threads(NUM_THREADS),private(i,j,k,tmp)
    for (k = 0; k < N; k++)
    {
        #pragma omp single
        {
            for (j = k + 1; j < N; j++)
                A[k][j] /= A[k][k];
            A[k][k] = 1.0;
        }
        #pragma omp for
        for (i = k + 1; i < N; i++)
        {
            float tmp[4] = { A[i][k],A[i][k],A[i][k],A[i][k] };
            __m128 arr_ik = _mm_loadu_ps(tmp);
            int num = k + 1;
            for (j = k + 1; j+4 <= N; j+=4,num=j)
            {
                __m128 arr_ij = _mm_loadu_ps(A[i] + j);
                __m128 arr_kj = _mm_loadu_ps(A[k] + j);
                arr_kj = _mm_mul_ps(arr_kj, arr_ik);
                arr_ij = _mm_sub_ps(arr_ij, arr_kj);
                _mm_storeu_ps(A[i] + j, arr_ij);
            }
            for (; j < N; j++)
                A[i][j] -= (A[i][k] * A[k][j]);
            A[i][k] = 0;
        }
    }
}
void Openmp_AVX_static()
{
    int i, j, k;
    #pragma omp parallel num_threads(NUM_THREADS),private(i,j,k,tmp)
    for (k = 0; k < N; k++)
    {
        //除法部分，一个线程执行
        #pragma omp single
        {
            for (j = k + 1; j < N; j++)
                A[k][j] /=A[k][k];
            A[k][k] = 1.0;
        }
        //消去部分，使用行划分
        #pragma omp for
        for (i = k + 1; i < N; i++)
        {
            float tmp[8] = { A[i][k],A[i][k],A[i][k],A[i][k],A[i][k],A[i][k],A[i][k],A[i][k] };
            __m256 arr_ik = _mm256_loadu_ps(tmp);
            int num = k + 1;
            for (j = k + 1; j+8 <= N; j+=8,num=j)
            {
                __m256 arr_ij = _mm256_loadu_ps(A[i] + j);
                __m256 arr_kj = _mm256_loadu_ps(A[k] + j);
                arr_kj = _mm256_mul_ps(arr_kj, arr_ik);
                arr_ij = _mm256_sub_ps(arr_ij, arr_kj);
                _mm256_storeu_ps(A[i] + j, arr_ij);
            }
            for (; j < N; j++)
                A[i][j] -= (A[i][k] * A[k][j]);
            A[i][k] = 0;
        }
    }
}
void Openmp_SSE_dynamic()
{
    int i, j, k;
    #pragma omp parallel num_threads(NUM_THREADS),private(i,j,k,tmp)
    for (k = 0; k < N; k++)
    {
        #pragma omp single
        {
            for (j = k + 1; j < N; j++)
                A[k][j] /= A[k][k];
            A[k][k] = 1.0;
        }
        #pragma omp for schedule(dynamic,3)
        for (i = k + 1; i < N; i++)
        {
            float tmp[4] = { A[i][k],A[i][k],A[i][k],A[i][k] };
            __m128 arr_ik = _mm_loadu_ps(tmp);
            int num = k + 1;
            for (j = k + 1; j+4 <= N; j+=4,num=j)
            {
                __m128 arr_ij = _mm_loadu_ps(A[i] + j);
                __m128 arr_kj = _mm_loadu_ps(A[k] + j);
                arr_kj = _mm_mul_ps(arr_kj, arr_ik);
                arr_ij = _mm_sub_ps(arr_ij, arr_kj);
                _mm_storeu_ps(A[i] + j, arr_ij);
            }
            for (; j < N; j++)
                A[i][j] -= (A[i][k] * A[k][j]);
            A[i][k] = 0;
        }
    }
}
void Openmp_AVX_dynamic()
{
    int i, j, k;
    #pragma omp parallel num_threads(NUM_THREADS),private(i,j,k,tmp)
    for (k = 0; k < N; k++)
    {
        //除法部分，一个线程执行
        #pragma omp single
        {
            for (j = k + 1; j < N; j++)
                A[k][j] /=A[k][k];
            A[k][k] = 1.0;
        }
        //消去部分，使用行划分
        #pragma omp for schedule(dynamic,3)
        for (i = k + 1; i < N; i++)
        {
            float tmp[8] = { A[i][k],A[i][k],A[i][k],A[i][k],A[i][k],A[i][k],A[i][k],A[i][k] };
            __m256 arr_ik = _mm256_loadu_ps(tmp);
            int num = k + 1;
            for (j = k + 1; j+8 <= N; j+=8,num=j)
            {
                __m256 arr_ij = _mm256_loadu_ps(A[i] + j);
                __m256 arr_kj = _mm256_loadu_ps(A[k] + j);
                arr_kj = _mm256_mul_ps(arr_kj, arr_ik);
                arr_ij = _mm256_sub_ps(arr_ij, arr_kj);
                _mm256_storeu_ps(A[i] + j, arr_ij);
            }
            for (; j < N; j++)
                A[i][j] -= (A[i][k] * A[k][j]);
            A[i][k] = 0;
        }
    }
}
void Openmp_SSE_guide()
{
    int i, j, k;
    #pragma omp parallel num_threads(NUM_THREADS),private(i,j,k,tmp)
    for (k = 0; k < N; k++)
    {
        #pragma omp single
        {
            for (j = k + 1; j < N; j++)
                A[k][j] /= A[k][k];
            A[k][k] = 1.0;
        }
        #pragma omp for schedule(guide)
        for (i = k + 1; i < N; i++)
        {
            float tmp[4] = { A[i][k],A[i][k],A[i][k],A[i][k] };
            __m128 arr_ik = _mm_loadu_ps(tmp);
            int num = k + 1;
            for (j = k + 1; j+4 <= N; j+=4,num=j)
            {
                __m128 arr_ij = _mm_loadu_ps(A[i] + j);
                __m128 arr_kj = _mm_loadu_ps(A[k] + j);
                arr_kj = _mm_mul_ps(arr_kj, arr_ik);
                arr_ij = _mm_sub_ps(arr_ij, arr_kj);
                _mm_storeu_ps(A[i] + j, arr_ij);
            }
            for (; j < N; j++)
                A[i][j] -= (A[i][k] * A[k][j]);
            A[i][k] = 0;
        }
    }
}
void Openmp_AVX_guide()
{
    int i, j, k;
    #pragma omp parallel num_threads(NUM_THREADS),private(i,j,k,tmp)
    for (k = 0; k < N; k++)
    {
        //除法部分，一个线程执行
        #pragma omp single
        {
            for (j = k + 1; j < N; j++)
                A[k][j] /=A[k][k];
            A[k][k] = 1.0;
        }
        //消去部分，使用行划分
        #pragma omp for schedule(dynami)
        for (i = k + 1; i < N; i++)
        {
            float tmp[8] = { A[i][k],A[i][k],A[i][k],A[i][k],A[i][k],A[i][k],A[i][k],A[i][k] };
            __m256 arr_ik = _mm256_loadu_ps(tmp);
            int num = k + 1;
            for (j = k + 1; j+8 <= N; j+=8,num=j)
            {
                __m256 arr_ij = _mm256_loadu_ps(A[i] + j);
                __m256 arr_kj = _mm256_loadu_ps(A[k] + j);
                arr_kj = _mm256_mul_ps(arr_kj, arr_ik);
                arr_ij = _mm256_sub_ps(arr_ij, arr_kj);
                _mm256_storeu_ps(A[i] + j, arr_ij);
            }
            for (; j < N; j++)
                A[i][j] -= (A[i][k] * A[k][j]);
            A[i][k] = 0;
        }
    }
}
int main()
{
    int epoch=2;
    long long begin, end, freq;
    double timer;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    init();
    timer=0;
    for (int i = 0; i < epoch; i++)
    {
        reset();
        QueryPerformanceCounter((LARGE_INTEGER*)&begin);
        Openmp_SSE_static();
        QueryPerformanceCounter((LARGE_INTEGER*)&end);
        timer += (end - begin) * 1000.0 / freq;
    }
    cout << "Openmp_SSE_static:  " << timer / epoch << "ms" << endl;
//    timer=0;
//    for (int i = 0; i < epoch; i++)
//    {
//        reset();
//        QueryPerformanceCounter((LARGE_INTEGER*)&begin);
//        Openmp_AVX_static();
//        QueryPerformanceCounter((LARGE_INTEGER*)&end);
//        timer += (end - begin) * 1000.0 / freq;
//    }
//    cout << "Openmp_AVX_static:  " << timer / epoch << "ms" << endl;
    timer=0;
    for (int i = 0; i < epoch; i++)
    {
        reset();
        QueryPerformanceCounter((LARGE_INTEGER*)&begin);
        Openmp_SSE_dynamic();
        QueryPerformanceCounter((LARGE_INTEGER*)&end);
        timer += (end - begin) * 1000.0 / freq;
    }
    cout << "Openmp_SSE_dynamic:  " << timer / epoch << "ms" << endl;
//    timer=0;
//    for (int i = 0; i < epoch; i++)
//    {
//        reset();
//        QueryPerformanceCounter((LARGE_INTEGER*)&begin);
//        Openmp_AVX_dynamic();
//        QueryPerformanceCounter((LARGE_INTEGER*)&end);
//        timer += (end - begin) * 1000.0 / freq;
//    }
//    cout << "Openmp_AVX_dynamic:  " << timer / epoch << "ms" << endl;
    timer=0;
    for (int i = 0; i < epoch; i++)
    {
        reset();
        QueryPerformanceCounter((LARGE_INTEGER*)&begin);
        Openmp_SSE_guide();
        QueryPerformanceCounter((LARGE_INTEGER*)&end);
        timer += (end - begin) * 1000.0 / freq;
    }
    cout << "Openmp_SSE_guide:  " << timer / epoch << "ms" << endl;
//    timer=0;
//    for (int i = 0; i < epoch; i++)
//    {
//        reset();
//        QueryPerformanceCounter((LARGE_INTEGER*)&begin);
//        Openmp_AVX_guide();
//        QueryPerformanceCounter((LARGE_INTEGER*)&end);
//        timer += (end - begin) * 1000.0 / freq;
//    }
//    cout << "Openmp_AVX_guide:  " << timer / epoch << "ms" << endl;
    return 0;
}
