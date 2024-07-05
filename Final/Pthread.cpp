#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <bitset>
#include <Windows.h>
#include <nmmintrin.h>
#include <immintrin.h>
#include <pthread.h>
using namespace std;
const int ColNum = 2362;
const int R_RowNum = 2362;
const int R_ColNum = (ColNum + 31) / 32;
const int E_RowNum = 453;
const int E_ColNum = (ColNum + 31) / 32;
unsigned int E[E_RowNum][E_ColNum] = {0}, R[R_RowNum][R_ColNum] = {0};
int First[E_RowNum] = {0};
int NUM_THREADS = 7;

sem_t sem_leader;
sem_t *sem_Next = new sem_t[NUM_THREADS - 1];

struct threadParam_t
{
    int t_id;
};
void Init()
{
    for (int i = 0; i < R_RowNum; i++)
    {
        for (int j = 0; j < R_ColNum; j++)
        {
            R[i][j] = 0;
        }
    }
    for (int i = 0; i < E_RowNum; i++)
    {
        for (int j = 0; j < E_ColNum; j++)
        {
            E[i][j] = 0;
        }
    }
}
void Init_E()
{
    ifstream file("2362_E.txt");
    string line;
    int index = 0;
    while (getline(file, line))
    {
        istringstream iss(line);
        unsigned int number;
        bool isFirst = false;
        while (iss >> number)
        {
            if (isFirst == false)
            {
                First[index] = number;
                isFirst = true;
            }
            int offset = number % 32;
            int post = number / 32;
            int temp = 1 << offset;
            E[index][E_ColNum - 1 - post] += temp;
        }
        index++;
    }
    file.close();
}
void Init_R()
{
    ifstream file("2362_R.txt");
    string line;
    int index = 0;
    while (getline(file, line))
    {
        istringstream iss(line);
        int number;
        bool isFirst = false;
        while (iss >> number)
        {
            if (isFirst == false) // 判断是不是首项
            {
                index = number; // 首项位置决定存放的行位置
                isFirst = true;
            }
            // 将数字存入对应行中
            int offset = number % 32;
            int post = number / 32;
            int temp = 1 << offset;
            R[index][R_ColNum - 1 - post] += temp;
        }
    }
    file.close();
}
void* ThreadFunc_SSE(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    int i = p->i;
    for (int i = ; i < E_RowNum; i++NUM_THREADS)
    {
        while (First[i] != -1)
        {
            bool isnull=true;
            for (int h = t_id; h < R_ColNum; h+=NUM_THREADS)
            {
                if (R[First[i]][h] != 0)
                {
                    isnull=false;
                    int num = 0;
                    for (int h = 0; h +4<= E_ColNum; h+=4,num=h)
                    {
                        __m128i vec_E = _mm_loadu_si128((__m128i*)(E[i] + h));
                        __m128i vec_R = _mm_loadu_si128((__m128i*)(R[First[i]] + h));
                        vec_E = _mm_xor_si128(vec_E, vec_R);
                        _mm_storeu_si128((__m128i*)(E[i]+h), vec_E);
                    }
                    for (int h = num; h < E_ColNum; h++)
                    {
                        E[i][h] = E[i][h] ^ R[First[i]][h];
                    }

                    int k;
                    for (k = 0; k < E_ColNum; k++)
                        if (E[i][k] != 0)
                            break;
                    if (k == E_ColNum)
                    {
                        First[i] = -1;
                        break;
                    }
                    unsigned int temp = E[i][k];
                    int j = 0;
                    while (temp != 0)
                    {
                        temp = temp >> 1;
                        j++;
                    }
                    First[i] = E_ColNum * 32 - (k + 1) * 32 + j - 1;
                }
            }
            pthread_barrier_wait(&barrier_nor);
            if(isnull&&t_id==0)
            {
                for (int j = 0; j < R_ColNum; j++)
                    R[First[i]][j] = E[i][j];
                break;
            }
            pthread_barrier_wait(&barrier_shengge);
        }
    }
    pthread_exit(NULL);
}
void sse() {
    pthread_barrier_init(&barrier_nor, NULL, NUM_THREADS);
    pthread_barrier_init(&barrier_shengge, NULL, NUM_THREADS);
    //创建线程
    pthread_t handles[NUM_THREADS];//创建对应的handle
    threadParam_t param[NUM_THREADS];//创建对应的线程数据结构


    for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
        param[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, threadFunc_Pthread, (void*)&param[t_id]);
    }

    for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
        pthread_join(handles[t_id], NULL);
    }

    pthread_barrier_destroy(&barrier_nor);
    pthread_barrier_destroy(&barrier_shengge);
}
// 线程函数定义
void *ThreadFunc_SSE_PRO(void *param)
{
    threadParam_t *p = (threadParam_t *)param;
    int t_id = p->t_id;
    bool isend = true;
    while (isend)
    {
        for (int i = t_id; i < E_RowNum; i += NUM_THREADS)
        {
            while (First[i] != -1)
            {
                bool isnull = true;
                for (int h = 0; h < R_ColNum; h++)
                {
                    if (R[First[i]][h] != 0)
                    {
                        isnull = false;
                        for (int j = 0; j < R_ColNum; j++)
                        {
                            E[i][j] = E[i][j] ^ R[First[i]][j];
                        }
                        int k;
                        for (k = 0; k < E_ColNum; k++)
                            if (E[i][k] != 0)
                                break;
                        if (k == E_ColNum)
                        {
                            First[i] = -1;
                            break;
                        }
                        unsigned int temp = E[i][k];
                        int j = 0;
                        while (temp != 0)
                        {
                            temp = temp >> 1;
                            j++;
                        }
                        First[i] = E_ColNum * 32 - (k + 1) * 32 + j - 1;
                    }
                }
                if (isnull)
                    break;
            }
        }
        if (t_id == 0)
        {
            for (int i = 0; i < NUM_THREADS - 1; i++)
                sem_wait(&sem_leader); // 等待其它 worker 完成消去
        }
        else
        {
            sem_post(&sem_leader);                // 通知 leader, 已完成升格任务
            sem_wait(&sem_Next[t_id - 1]); // 等待通知，进入下一轮
        }
        if (t_id == 0)
        {
            for (int i = 0; i < R_RowNum; i++)
            {
                int tmp = First[i];
                if (tmp == -1)
                    continue;
                if (R[tmp][E_ColNum] == 0)
                {
                    for (int k = 0; k < R_RowNum; k++)
                        R[tmp][k] = E[i][k];
                    E[i][E_ColNum] = -1;
                    isend = true;
                }
            }
        }
        if (t_id == 0)
        {
            for (int i = 0; i < NUM_THREADS - 1; ++i)
                sem_post(&sem_Next[i]); // 通知其它 worker 进入下一轮
        }
    }
    pthread_exit(NULL);
}
void Thread_SSE_PRO()
{
    // 初始化信号量
    sem_init(&sem_leader, 0, 0);
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    {
        sem_init(&sem_Divsion[t_id], 0, 0);
        sem_init(&sem_Elimination[t_id], 0, 0);
    }
    // 创建线程
    pthread_t handles[NUM_THREADS];   // 创建对应的 Handle
    threadParam_t param[NUM_THREADS]; // 创建对应的线程数据结构
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    {
        param[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, ThreadFunc_SSE_PRO, (void *)&param[t_id]);
    }
    // 等待回收线程
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        pthread_join(handles[t_id], NULL);
    // 销毁信号量
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
    long long head, tail, freq;
    long double timer = 0;
    for (int i = 0; i < epoch; i++)
    {
        Init();
        Init_E();
        Init_R();
        QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
        QueryPerformanceCounter((LARGE_INTEGER *)&head);
        Thread_SSE();
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        timer += (tail - head) * 1000.0 / freq;
    }
    cout << endl
         << timer / epoch << "ms" << endl;
    timer = 0;
    for (int i = 0; i < epoch; i++)
    {
        Init();
        Init_E();
        Init_R();
        QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
        QueryPerformanceCounter((LARGE_INTEGER *)&head);
        Thread_SSE_PRO();
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        timer += (tail - head) * 1000.0 / freq;
    }
    cout << endl
         << timer / epoch << "ms" << endl;
    return 0;
}