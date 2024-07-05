#include<iostream>
#include<fstream>
#include<sstream>
#include<cmath>
#include<bitset>
#include<Windows.h>
#include <nmmintrin.h>
#include <immintrin.h>
using namespace std;
//const int ColNum = 2362;
//const int R_RowNum = 2362;
//const int R_ColNum = (ColNum+31)/ 32;
//const int E_RowNum = 453;
//const int E_ColNum = (ColNum+31)/ 32;
//unsigned int E[E_RowNum][E_ColNum]= {0},R[R_RowNum][R_ColNum]= {0};
//int First[E_RowNum]= {0};
const int ColNum = 2362;
const int R_RowNum = 2362;
const int R_ColNum = (ColNum+31)/ 32;
const int E_RowNum = 453;
const int E_ColNum = (ColNum+31)/ 32;
unsigned int E[E_RowNum][E_ColNum]= {0},R[R_RowNum][R_ColNum]= {0};
int First[E_RowNum]= {0};
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
    ifstream file("C:\\Users\\86178\\Desktop\\Parallel\\Para_Code\\SIMD\\special_Gauss\\data\\2362_E.txt");
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
    ifstream file("C:\\Users\\86178\\Desktop\\Parallel\\Para_Code\\SIMD\\special_Gauss\\data\\2362_R.txt");
    string line;
    int index = 0;
    while (getline(file, line))
    {
        istringstream iss(line);
        int number;
        bool isFirst = false;
        while (iss >> number)
        {
            if (isFirst == false)   //�ж��ǲ�������
            {
                index = number; //����λ�þ�����ŵ���λ��
                isFirst = true;
            }
            //�����ִ����Ӧ����
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
    for (int i = 0; i < E_RowNum; i++)
    {
        while (First[i] != -1)
        {
            bool isnull=true;
            for (int h = 0; h < R_ColNum; h++)
            {
                if (R[First[i]][h] != 0)
                {
                    isnull=false;
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
            if(isnull)
            {
                for (int j = 0; j < R_ColNum; j++)
                    R[First[i]][j] = E[i][j];
                break;
            }
        }
    }
}
void sse()
{
    for (int i = 0; i < E_RowNum; i++)
    {
        while (First[i] != -1)
        {
            bool isnull=true;
            for (int h = 0; h < R_ColNum; h++)
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
            if(isnull)
            {
                for (int j = 0; j < R_ColNum; j++)
                    R[First[i]][j] = E[i][j];
                break;
            }
        }
    }
}
void avx()
{
    for (int i = 0; i < E_RowNum; i++)
    {
        while (First[i] != -1)
        {
            bool isnull=true;
            for (int h = 0; h < R_ColNum; h++)
            {
                if (R[First[i]][h] != 0)
                {
                    isnull=false;
                    int num = 0;
                    for (int h = 0; h +8<= E_ColNum; h+=8,num=h)
                    {
                        __m256i vec_E = _mm256_loadu_si256((__m256i*)(E[i] + h));
                        __m256i vec_R = _mm256_loadu_si256((__m256i*)(R[First[i]] + h));
                        vec_E = _mm256_xor_si256(vec_E, vec_R);
                        _mm256_storeu_si256((__m256i*)(E[i] + h), vec_E);
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
            if(isnull)
            {
                for (int j = 0; j < R_ColNum; j++)
                    R[First[i]][j] = E[i][j];
                break;
            }
        }
    }
}

int main()
{
    int epoch = 10;
    long long head, tail, freq;
    long double timer=0;
    for (int i = 0; i < epoch; i++)
    {
        Init();
        Init_E();
        Init_R();
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        normal();
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        timer += (tail - head) * 1000.0 / freq;
    }
    cout<<endl  << timer / epoch << "ms" << endl;
    timer=0;
    for (int i = 0; i < epoch; i++)
    {
        Init();
        Init_E();
        Init_R();
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        sse();
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        timer += (tail - head) * 1000.0 / freq;
    }
    cout<<endl  << timer / epoch << "ms" << endl;
    timer=0;
    for (int i = 0; i < epoch; i++)
    {
        Init();
        Init_E();
        Init_R();
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        avx();
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        timer += (tail - head) * 1000.0 / freq;
    }
    cout<<endl  << timer / epoch << "ms" << endl;
    return 0;
}



