
#include<iostream>
#include<windows.h>
#include<nmmintrin.h>
#include<immintrin.h>
#include<pthread.h>
#include<semaphore.h>
using namespace std;
const int N = 4000;
const int NUM_THREADS = 7;
float A[N][N];
float matrix[N][N];
struct threadParam_t
{
    int k; //消去的轮次
    int t_id; // 线程 id
};
//信号量
sem_t sem_leader;
sem_t sem_Divsion[NUM_THREADS - 1];
sem_t sem_Elimination[NUM_THREADS - 1];
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
        // t_id 为 0 的线程做除法操作，其它工作线程先等待
        // 这里只采用了一个工作线程负责除法操作，同学们可以尝试采用多个工作线程完成除法操作
        // 比信号量更简洁的同步方式是使用 barrier
        if (t_id == 0)
        {
            for (int j = k + 1; j < N; j++)
                A[k][j] /= A[k][k];
            A[k][k] = 1.0;
        }
        else
            sem_wait(&sem_Divsion[t_id - 1]); //阻塞，等待完成除法操作
        // t_id 为 0 的线程唤醒其它工作线程，进行消去操作
        if (t_id == 0)
            for (int i = 0; i < NUM_THREADS - 1; i++)
                sem_post(&sem_Divsion[i]);
        //循环划分任务（可以尝试多种任务划分方式）
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
        //等待进入下一轮循环
        if (t_id == 0)
        {
            for (int i = 0; i < NUM_THREADS - 1; i++)
                sem_wait(&sem_leader);// 等待其它 worker 完成消去
            for (int i = 0; i < NUM_THREADS - 1; i++)
                sem_post(&sem_Elimination[i]); // 通知其它 worker 进入下一轮
        }
        else
        {
            sem_post(&sem_leader);// 通知 leader, 已完成消去任务
            sem_wait(&sem_Elimination[t_id - 1]); // 等待通知，进入下一轮
        }

    }
    pthread_exit(NULL);
    return 0;
}
void Thread_SSE()
{
    //初始化信号量
    sem_init(&sem_leader, 0, 0);
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    {
        sem_init(&sem_Divsion[t_id], 0, 0);
        sem_init(&sem_Elimination[t_id], 0, 0);
    }
    //创建线程
    pthread_t handles[NUM_THREADS]; // 创建对应的 Handle
    threadParam_t param[NUM_THREADS]; // 创建对应的线程数据结构
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    {
        param[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, ThreadFunc_SSE, (void*)&param[t_id]);
    }
    //等待回收线程
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        pthread_join(handles[t_id], NULL);
    //销毁信号量
    sem_destroy(&sem_leader);
    for (int i = 0; i < NUM_THREADS - 1; i++)
    {
        sem_destroy(&sem_Divsion[i]);
        sem_destroy(&sem_Elimination[i]);
    }
}

//线程函数定义
void* ThreadFunc_SSE_COL(void* param)
{
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    for (int k = 0; k < N; k++)
    {
        // t_id 为 0 的线程做除法操作，其它工作线程先等待
        // 这里只采用了一个工作线程负责除法操作，同学们可以尝试采用多个工作线程完成除法操作
        // 比信号量更简洁的同步方式是使用 barrier
        if (t_id == 0)
        {
            for (int j = k + 1; j < N; j++)
                A[k][j] /= A[k][k];
            A[k][k] = 1.0;
        }
        else
            sem_wait(&sem_Divsion[t_id - 1]); //阻塞，等待完成除法操作
        // t_id 为 0 的线程唤醒其它工作线程，进行消去操作
        if (t_id == 0)
            for (int i = 0; i < NUM_THREADS - 1; i++)
                sem_post(&sem_Divsion[i]);
        //循环划分任务（可以尝试多种任务划分方式）
        for (int i = k + 1; i < N; i += 1)
        {
            //消去
            float tmp[4] = { A[i][k],A[i][k],A[i][k],A[i][k] };
            __m128 arr_ik = _mm_loadu_ps(tmp);
            int num = k + 1;
            int j;
            for (j = k + 1+t_id; j + 4+NUM_THREADS <= N; j += 4+NUM_THREADS, num = j)
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
        //等待进入下一轮循环
        if (t_id == 0)
        {
            for (int i = 0; i < NUM_THREADS - 1; i++)
                sem_wait(&sem_leader);// 等待其它 worker 完成消去
            for (int i = 0; i < NUM_THREADS - 1; i++)
                sem_post(&sem_Elimination[i]); // 通知其它 worker 进入下一轮
        }
        else
        {
            sem_post(&sem_leader);// 通知 leader, 已完成消去任务
            sem_wait(&sem_Elimination[t_id - 1]); // 等待通知，进入下一轮
        }

    }
    pthread_exit(NULL);
    return 0;
}
void Thread_SSE_COL()
{
    //初始化信号量
    sem_init(&sem_leader, 0, 0);
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    {
        sem_init(&sem_Divsion[t_id], 0, 0);
        sem_init(&sem_Elimination[t_id], 0, 0);
    }
    //创建线程
    pthread_t handles[NUM_THREADS]; // 创建对应的 Handle
    threadParam_t param[NUM_THREADS]; // 创建对应的线程数据结构
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    {
        param[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, ThreadFunc_SSE_COL, (void*)&param[t_id]);
    }
    //等待回收线程
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        pthread_join(handles[t_id], NULL);
    //销毁信号量
    sem_destroy(&sem_leader);
    for (int i = 0; i < NUM_THREADS - 1; i++)
    {
        sem_destroy(&sem_Divsion[i]);
        sem_destroy(&sem_Elimination[i]);
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
        Thread_SSE();
        QueryPerformanceCounter((LARGE_INTEGER*)&end);
        timer += (end - begin) * 1000.0 / freq;
    }
    cout << " Pthread_SSE:  " << timer / epoch << " ms" << endl;
    timer = 0;
    for (int i = 0; i < epoch; i++)
    {
        reset();
        QueryPerformanceCounter((LARGE_INTEGER*)&begin);
        Thread_SSE_COL();
        QueryPerformanceCounter((LARGE_INTEGER*)&end);
        timer += (end - begin) * 1000.0 / freq;
    }
    cout << " Pthread_SSE_COL:  " << timer / epoch << " ms" << endl;
    return 0;
}
