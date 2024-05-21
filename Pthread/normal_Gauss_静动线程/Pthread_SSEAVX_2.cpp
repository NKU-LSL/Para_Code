#include<iostream>
#include<windows.h>
#include<nmmintrin.h>
#include<immintrin.h>
#include<pthread.h>
#include<semaphore.h>
using namespace std;
const int N = 4096;
const int NUM_THREADS = 7;
float A[N][N];
float matrix[N][N];
struct threadParam_t
{
    int k; //消去的轮次
    int t_id; // 线程 id
};
//信号量
sem_t sem_main;
sem_t sem_workerstart[NUM_THREADS];
sem_t sem_workerend[NUM_THREADS];
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

//线程函数定义
void* ThreadFunc_SSE(void* param)
{
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    for (int k = 0; k < N; k++)
    {
        sem_wait(&sem_workerstart[t_id]); //阻塞，等待主线程完成除法操作（操作自己专属的信号量）
        //循环划分任务
        for (int i = k + 1 + t_id; i < N; i += NUM_THREADS)
        {
            //消去
            float tmp[4] = { A[i][k],A[i][k],A[i][k],A[i][k] };
            __m128 arr_ik = _mm_loadu_ps(tmp);
            int num = k + 1;
            int j;
            for (j = k + 1; j + 4 <= N; j += 4, num = j)
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
        sem_post(&sem_main); //唤醒主线程
        sem_wait(&sem_workerend[t_id]); //阻塞，等待主线程唤醒进入下一轮
    }
    pthread_exit(NULL);
    return 0;
}
void Thread_SSE()
{
    //初始化信号量
    sem_init(&sem_main, 0, 0);
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    {
        sem_init(&sem_workerstart[t_id], 0, 0);
        sem_init(&sem_workerend[t_id], 0, 0);
    }
    //创建线程
    pthread_t handles[NUM_THREADS]; // 创建对应的 Handle
    threadParam_t param[NUM_THREADS]; // 创建对应的线程数据结构
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    {
        param[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, ThreadFunc_SSE, (void*)&param[t_id]);
    }
    for (int k = 0; k < N; k++)
    {
        //主线程做除法操作
        for (int j = k + 1; j < N; j++)
            A[k][j] /= A[k][k];
        A[k][k] = 1.0;
        //开始唤醒工作线程
        for (int t_id = 0; t_id < NUM_THREADS; t_id++)
            sem_post(&sem_workerstart[t_id]);
        //主线程睡眠（等待所有的工作线程完成此轮消去任务）
        for (int t_id = 0; t_id < NUM_THREADS; t_id++)
            sem_wait(&sem_main);
        // 主线程再次唤醒工作线程进入下一轮次的消去任务
        for (int t_id = 0; t_id < NUM_THREADS; t_id++)
            sem_post(&sem_workerend[t_id]);
    }
    //等待回收工作线程
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        pthread_join(handles[t_id], NULL);
    //销毁信号量
    sem_destroy(&sem_main);
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) 
    {
        sem_destroy(&sem_workerstart[t_id]);
        sem_destroy(&sem_workerend[t_id]);
    }
}

//线程函数定义
void* ThreadFunc_AVX(void* param)
{
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    for (int k = 0; k < N; k++)
    {
        sem_wait(&sem_workerstart[t_id]); //阻塞，等待主线程完成除法操作（操作自己专属的信号量）
        //循环划分任务
        for (int i = k + 1 + t_id; i < N; i += NUM_THREADS)
        {
            //消去
            float tmp[8] = { A[i][k],A[i][k],A[i][k],A[i][k],A[i][k],A[i][k],A[i][k],A[i][k] };
            __m256 arr_ik = _mm256_loadu_ps(tmp);
            int num = k + 1;
            int j;
            for (j = k + 1; j + 8 <= N; j += 8, num = j)
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
        sem_post(&sem_main); //唤醒主线程
        sem_wait(&sem_workerend[t_id]); //阻塞，等待主线程唤醒进入下一轮
    }
    pthread_exit(NULL);
    return 0;
}
void Thread_AVX()
{
    //初始化信号量
    sem_init(&sem_main, 0, 0);
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    {
        sem_init(&sem_workerstart[t_id], 0, 0);
        sem_init(&sem_workerend[t_id], 0, 0);
    }
    //创建线程
    pthread_t handles[NUM_THREADS]; // 创建对应的 Handle
    threadParam_t param[NUM_THREADS]; // 创建对应的线程数据结构
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    {
        param[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, ThreadFunc_AVX, (void*)&param[t_id]);
    }
    for (int k = 0; k < N; k++)
    {
        //主线程做除法操作
        for (int j = k + 1; j < N; j++)
            A[k][j] /= A[k][k];
        A[k][k] = 1.0;
        //开始唤醒工作线程
        for (int t_id = 0; t_id < NUM_THREADS; t_id++)
            sem_post(&sem_workerstart[t_id]);
        //主线程睡眠（等待所有的工作线程完成此轮消去任务）
        for (int t_id = 0; t_id < NUM_THREADS; t_id++)
            sem_wait(&sem_main);
        // 主线程再次唤醒工作线程进入下一轮次的消去任务
        for (int t_id = 0; t_id < NUM_THREADS; t_id++)
            sem_post(&sem_workerend[t_id]);
    }
    //等待回收工作线程
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        pthread_join(handles[t_id], NULL);
    //销毁信号量
    sem_destroy(&sem_main);
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) 
    {
        sem_destroy(&sem_workerstart[t_id]);
        sem_destroy(&sem_workerend[t_id]);
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
    cout << " normal:  " << timer / epoch << "ms" << endl;
    timer = 0;
    for (int i = 0; i < epoch; i++)
    {
        reset();
        QueryPerformanceCounter((LARGE_INTEGER*)&begin);
        Thread_SSE();
        QueryPerformanceCounter((LARGE_INTEGER*)&end);
        timer += (end - begin) * 1000.0 / freq;
    }
    cout << " Pthread_SSE:  " << timer / epoch << "ms" << endl;
    timer = 0;
    for (int i = 0; i < epoch; i++)
    {
        reset();
        QueryPerformanceCounter((LARGE_INTEGER*)&begin);
        Thread_AVX();
        QueryPerformanceCounter((LARGE_INTEGER*)&end);
        timer += (end - begin) * 1000.0 / freq;
    }
    cout << " Pthread_AVX:  " << timer / epoch << "ms" << endl;
    return 0;
}
