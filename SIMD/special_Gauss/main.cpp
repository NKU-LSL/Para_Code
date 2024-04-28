#include<iostream>
#include<fstream>
#include<sstream>
#include<cmath>
#include<bitset>
#include <nmmintrin.h>
#include <immintrin.h>
#include <windows.h>
#include <pmmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <tmmintrin.h>
using namespace std;
const int ColNum = 2362;
const int R_RowNum = ColNum;
const int R_ColNum = ColNum/ 32 + (ColNum%32>0);
const int E_RowNum = 453;
const int E_ColNum = ColNum/ 32 + (ColNum%32>0);
unsigned int E[E_RowNum][E_ColNum]= {0},R[R_RowNum][R_ColNum]= {0};
int First[E_RowNum]= {0};
void Init_E()
{
    ifstream file("C:\\Users\\86178\\Desktop\\Parallel\\SIMD\\special_Gauss\\1_A.txt");
    string line;
    int index = 0;
    while (getline(file, line))
    {
        istringstream iss(line);
        unsigned int number;
        bool isFirst = false;
        while (iss >> number)
        {
            if (isFirst == false)   //判断是不是首项
            {
                First[index] = number;
                isFirst = true;
            }
            //将数字存入矩阵中
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
    ifstream file("C:\\Users\\86178\\Desktop\\Parallel\\SIMD\\special_Gauss\\1_B.txt");
    string line;
    int index = 0;
    while (getline(file, line))
    {
        istringstream iss(line);
        int number;
        bool isFirst = false;
        while (iss >> number)
        {
            if (isFirst == false)   //判断是不是首项
            {
                index = number; //首项位置决定存放的行位置
                isFirst = true;
            }
            //将数字存入对应行中
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
    for (int i = 0; i < Ma1_RowNum; i++)
    {
        while (First[i] != -1)
        {
            for (int i = 0; i < Ma2_ColNum; i++)
            {
                if (Ma2[First[i]][i] != 0)
                {
                    for (int j = 0; j < Ma2_ColNum; j++)
                    {
                        Ma2[First[i]][j] = E[i][j];
                    }

                }
                else
                {
                    for (int j = 0; j < Ma1_ColNum; j++)
                    {
                        Ma1[i][j] = Ma1[i][j] ^ Ma2[First[i]][j];
                    }
                    int j;
                    for (j = 0; j < Ma1_ColNum; j++)
                    {
                        if (Ma1[i][j] != 0)
                        {
                            break;
                        }
                    }
                    if (j == Ma1_ColNum)
                    {
                        First[i] = -1;
                        return;
                    }
                    unsigned int temp = Ma2[i][j];
                    int k = 0;
                    while (temp != 0)
                    {
                        temp = temp >> 1;
                        k++;
                    }
                    First[i] = Ma1_ColNum * 32 - (j + 1) * 32 + k - 1;
                }
            }
        }
    }
}
void sse()
{
    for (int i = 0; i < Ma1_RowNum; i++)
    {
        while (First[i] != -1)
        {
            for (int i = 0; i < Ma2_ColNum; i++)
            {
                if (Ma2[First[i]][i] != 0)
                {
                    for (int j = 0; j < Ma2_ColNum; j++)
                    {
                        Ma2[First[i]][j] = E[i][j];
                    }

                }
                else
                {
                    int num = 0;
                    for (int i = 0; i < E_ColNum - 3; i+=4,num=i)
                    {
                        __m128i t1 = _mm_loadu_si128((__m128i*)(Ma1[i] + k));
                        __m128i t2 = _mm_loadu_si128((__m128i*)(Ma2[First] + k));
                        t1 = _mm_xor_si128(t1, t2);
                        _mm_storeu_si128((__m128i*)(E[i]+k), t1);
                    }
                    for (int j = 0; j < Ma1_ColNum; j++)
                    {
                        Ma1[i][j] = Ma1[i][j] ^ Ma2[First[i]][j];
                    }
                    int j;
                    for (j = 0; j < Ma1_ColNum; j++)
                    {
                        if (Ma1[i][j] != 0)
                        {
                            break;
                        }
                    }
                    if (j == Ma1_ColNum)
                    {
                        First[i] = -1;
                        return;
                    }
                    unsigned int temp = Ma2[i][j];
                    int k = 0;
                    while (temp != 0)
                    {
                        temp = temp >> 1;
                        k++;
                    }
                    First[i] = Ma1_ColNum * 32 - (j + 1) * 32 + k - 1;
                }
            }
        }
    }
}
void avx()
{
    for (int i = 0; i < Ma1_RowNum; i++)
    {
        while (First[i] != -1)
        {
            for (int i = 0; i < Ma2_ColNum; i++)
            {
                if (Ma2[First[i]][i] != 0)
                {
                    for (int j = 0; j < Ma2_ColNum; j++)
                    {
                        Ma2[First[i]][j] = E[i][j];
                    }

                }
                else
                {
                    int num = 0;
                    for (int i = 0; i +4<= E_ColNum; i+=4,num=i)
                    {
                        __m256i t1 = _mm256_loadu_si256((__mm256i*)(Ma1[i] + k));
                        __m256i t2 = _mm256_loadu_si256((__m256i*)(Ma2[First] + k));
                        t1 = _mm256_xor_si256(t1, t2);
                        _mm256_storeu_si256((__m128i*)(E[i]+k), t1);
                    }
                    for (int j = 0; j < Ma1_ColNum; j++)
                    {
                        Ma1[i][j] = Ma1[i][j] ^ Ma2[First[i]][j];
                    }
                    int j;
                    for (j = 0; j < Ma1_ColNum; j++)
                    {
                        if (Ma1[i][j] != 0)
                        {
                            break;
                        }
                    }
                    if (j == Ma1_ColNum)
                    {
                        First[i] = -1;
                        return;
                    }
                    unsigned int temp = Ma2[i][j];
                    int k = 0;
                    while (temp != 0)
                    {
                        temp = temp >> 1;
                        k++;
                    }
                    First[i] = Ma1_ColNum * 32 - (j + 1) * 32 + k - 1;
                }
            }
        }
    }
}
int main()
{
    int epoch = 10;
    long long head, tail, freq;
    long double timer=0;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    Init_E();
    Init_R();
    cout<<E_ColNum<<" "<<E_RowNum<<" "<<R_ColNum<<" "<<R_RowNum<<endl;
    //串行算法计时
    for (int i = 0; i < epoch; i++)
    {
        Init_E();
        Init_R();
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        serial();
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        timer += (tail - head) * 1000.0 / freq;
    }
    cout  << timer / epoch << "ms" << endl;
    return 0;
}
