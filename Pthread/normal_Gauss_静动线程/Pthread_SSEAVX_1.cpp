#include<iostream>
#include<windows.h>
#include<nmmintrin.h>
#include<immintrin.h>
#include<pthread.h>
using namespace std;
const int N = 4096;
const int worker_count  = 7;
float A[N][N];
float matrix[N][N];
struct threadParam_t
{
    int k; //消去的轮次
    int t_id; // 线程 id
};
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

void* ThreadFunc_SSE(void* param)
{
    threadParam_t* p = (threadParam_t*)param;
    int k = p->k;//消去的轮次
    int t_id = p->t_id;//线程编号
    int i = k + t_id + 1; //获取自己的计算任务
    for (i = k + 1 + t_id; i < N; i += worker_count)
    {
        float tmp[4] = { A[i][k],A[i][k],A[i][k],A[i][k] };
        __m128 arr_ik = _mm_loadu_ps(tmp);
        int num = k + 1;
        int j;
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
    pthread_exit(NULL);
    return 0;
}
void Thread_SSE()
{
    for (int k = 0; k < N; k++)
    {
        //主线程做除法操作
        for (int j = k + 1; j < N; j++)
        {
            A[k][j] /= A[k][k];
        }
        A[k][k] = 1.0;
        //创建子线程，进行消去操作
        pthread_t* handles = new pthread_t[worker_count ]; //创建对应句柄
        threadParam_t* param = new threadParam_t[worker_count ]; //创建对应参数
        //分配任务
        for (int t_id = 0; t_id < worker_count; t_id++)
        {
            param[t_id].k = k;
            param[t_id].t_id = t_id;
        }
        //创建线程
        for (int t_id = 0; t_id < worker_count; t_id++)
        {
            pthread_create(&handles[t_id], NULL, ThreadFunc_SSE, (void*)&param[t_id]);
        }
        //主线程等待回收所有子线程
        for (int t_id = 0; t_id < worker_count; t_id++)
        {
            pthread_join(handles[t_id], NULL);
        }
        //释放分配的空间
        delete[]handles;
        delete[]param;
    }
}
void* ThreadFunc_AVX(void* param)
{
    threadParam_t* p = (threadParam_t*)param;
    int k = p->k;//消去的轮次
    int t_id = p->t_id;//线程编号
    int i = k + t_id + 1; //获取自己的计算任务

    for (i = k + 1 + t_id; i < N; i += worker_count)
    {
        float tmp[8] = { A[i][k],A[i][k],A[i][k],A[i][k],A[i][k],A[i][k],A[i][k],A[i][k] };
        __m256 arr_ik = _mm256_loadu_ps(tmp);
        int num = k + 1;
        int j;
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
    pthread_exit(NULL);
    return 0;
}
void Thread_AVX()
{
    for (int k = 0; k < N; k++)
    {
        //主线程做除法操作
        for (int j = k + 1; j < N; j++)
        {
            A[k][j] /= A[k][k];
        }
        A[k][k] = 1.0;
        //创建子线程，进行消去操作
        pthread_t* handles = new pthread_t[worker_count]; //创建对应句柄
        threadParam_t* param = new threadParam_t[worker_count]; //创建对应参数
        //分配任务
        for (int t_id = 0; t_id < worker_count; t_id++)
        {
            param[t_id].k = k;
            param[t_id].t_id = t_id;
        }
        //创建线程
        for (int t_id = 0; t_id < worker_count; t_id++)
        {
            pthread_create(&handles[t_id], NULL, ThreadFunc_AVX, (void*)&param[t_id]);
        }
        //主线程等待回收所有子线程
        for (int t_id = 0; t_id < worker_count; t_id++)
        {
            pthread_join(handles[t_id], NULL);
        }
        //释放分配的空间
        delete[]handles;
        delete[]param;
    }
}
int main()
{
    int epoch=10;
    long long begin, end, freq;
    double timer;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    init();
    timer=0;
    for (int i = 0; i < epoch; i++)
    {
        reset();
        QueryPerformanceCounter((LARGE_INTEGER*)&begin);
        normal();
        QueryPerformanceCounter((LARGE_INTEGER*)&end);
        timer += (end - begin) * 1000.0 / freq;
    }
    cout << "normal:  " << timer / epoch << "ms" << endl;
    timer=0;
    for (int i = 0; i < epoch; i++)
    {
        reset();
        QueryPerformanceCounter((LARGE_INTEGER*)&begin);
        Thread_SSE();
        QueryPerformanceCounter((LARGE_INTEGER*)&end);
        timer += (end - begin) * 1000.0 / freq;
    }
    cout << "Pthread_SSE:  " << timer / epoch << "ms" << endl;
    timer=0;
    for (int i = 0; i < epoch; i++)
    {
        reset();
        QueryPerformanceCounter((LARGE_INTEGER*)&begin);
        Thread_AVX();
        QueryPerformanceCounter((LARGE_INTEGER*)&end);
        timer += (end - begin) * 1000.0 / freq;
    }
    cout << "Pthread_AVX:  " << timer / epoch << "ms" << endl;
    return 0;
}
