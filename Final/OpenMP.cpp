#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <bitset>
#include <Windows.h>
#include <nmmintrin.h>
#include <immintrin.h>
#include <omp.h>
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
void normal()
{
    bool isend = true;
    while (isend)
    {
        for (int i = 0; i < E_RowNum; i++)
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
        isend = false;
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
}
void OpenMP(int rank, int size)
{
    #pragma omp parallel num_threads(thread_count) 
    for (int i = 0; i < E_RowNum; i++)
    {
        while (First[i] != -1)
        {
            bool isnull = true;
            for (int h = 0; h < R_ColNum; h++)
            {   
                if (R[First[i]][h] != 0)
                {
                    #pragma omp for schedule(static) 
                    __m128i t1, t2;
                    isnull = false;
                    int h = 0;
                    for (; h + 4 < R_ColNum; h += 4)
                    {
                        t1 = _mm_loadu_si128((__m128i*) & (E[i][h]));
                        t2 = _mm_loadu_si128((__m128i*) & (R[First[i]][h]));
                        t1 = _mm_xor_si128(t1, t2);
                        _mm_store_si128((__m128i*) & (E[i][h]), t1);
                    }
                    for (; h < R_ColNum; h++)
                        E[i][h] = E[i][h] ^ R[First[i]][h];
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
            {
                for (int j = 0; j < R_ColNum; j++)
                    R[First[i]][j] = E[i][j];
                break;
            }
        }
    }
}
// 线程函数定义
void Openmp_PRO()
{
#pragma omp parallel num_threads(NUM_THREADS)
    bool isend = true;
    while (isend)
    {
#pragma omp for schedule(static)
        for (int i = 0; i < E_RowNum; i += NUM_THREADS)
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
#pragma omp single
        isend=false;
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
        Openmp();
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
        Openmp_PRO();
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        timer += (tail - head) * 1000.0 / freq;
    }
    cout << endl
         << timer / epoch << "ms" << endl;
    return 0;
}