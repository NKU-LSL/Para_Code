#include <iostream>
#include <windows.h>
#include <nmmintrin.h>
#include <immintrin.h>
#include <mpi.h>
#include <omp.h>
#include<fstream>
#include<sstream>
#include<cmath>
#include<bitset>
#include<string>
using namespace std;
const int thread_count = 4;
const int ColNum = 2362;
const int R_RowNum = 2362;
const int R_ColNum = (ColNum + 31) / 32;
const int E_RowNum = 453;
const int E_ColNum = (ColNum + 31) / 32;
unsigned int E[E_RowNum][E_ColNum] = { 0 }, R[R_RowNum][R_ColNum] = { 0 };
int First[E_RowNum] = { 0 };
void print()
{
    for (int i = 0; i < E_RowNum; i++)
        for (int j = 0; j < E_ColNum; j++)
            if (E[i][j] != 0)
            {
                cout << E[i][j] << endl;
                return;
            }
}
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
        stringstream iss(line);
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
            {
                for (int j = 0; j < R_ColNum; j++)
                    R[First[i]][j] = E[i][j];
                break;
            }
        }
    }
}

void MPI_OpenMP(int rank, int size)
{
    #pragma omp parallel num_threads(NUM_THREADS)
    bool isend = true;
    while (isend)
    {
        for (int i = 0; i < E_RowNum; i++)
        {
            while (First[i] != -1)
            {
                #pragma omp for schedule(static)
                bool isnull=true;
                for (int h = 0; h < R_ColNum; h++)
                {
                    if (R[First[i]][h] != 0)
                    {
                        isnull=false;   
                        for (int j = 0; j < R_ColNum; j++)
                            E[i][j] = E[i][j] ^ R[First[i]][j];
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
                    break;
            }
        }
        #pragma omp single
        isend=false;
        for (int i = 0; i < R_RowNum; i++)
        {
            if (First[i] == -1)
                continue;
            if (R[First[i]][E_ColNum] == 0)
            {
                for (int k = 0; k < R_RowNum; k++)
                    R[First[i]][k] = E[i][k];
                E[i][E_ColNum] = -1;
                isend = true;
            }
        }
    }
}
void Main_OpenMP()
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int block = E_RowNum / size;
    int remain = E_RowNum % size;
    if (rank == 0)
    {
        for (int i = 1; i < size; i++)
        {
            int start_row = i * block;
            int num_rows = (i == size - 1) ? block + remain : block;
            MPI_Send(&E[start_row][0], num_rows * E_ColNum, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&First[start_row], num_rows, MPI_INT, i, 1, MPI_COMM_WORLD);
        }
    }
    else
    {
        int start_row = rank * block;
        int num_rows = (rank == size - 1) ? block + remain : block;
        MPI_Recv(&E[start_row][0], num_rows * E_ColNum, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&First[start_row], num_rows, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_OpenMP(rank, size);
    if (rank == 0)
    {
        for (int i = 1; i < size; i++)
        {
            int start_row = i * block;
            int num_rows = (i == size - 1) ? block + remain : block;
            MPI_Recv(&E[start_row][0], num_rows * E_ColNum, MPI_INT, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&First[start_row], num_rows, MPI_INT, i, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    else
    {
        int start_row = rank * block;
        int num_rows = (rank == size - 1) ? block + remain : block;
        MPI_Send(&E[start_row][0], num_rows * E_ColNum, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&First[start_row], num_rows, MPI_INT, 0, 3, MPI_COMM_WORLD);
    }
}
int main()
{
    int epoch = 1;
    long long head, tail, freq;
    long double timer = 0;
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
    print();
    cout << "normal:" << timer / epoch << "ms" << endl;
    timer = 0; MPI_Init(NULL, NULL);
    for (int i = 0; i < epoch; i++)
    {
        Init();
        Init_E();
        Init_R();
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        Main_OpenMP();
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        timer += (tail - head) * 1000.0 / freq;
    }
    print();
    cout << "Main_OpenMP:" << timer / epoch << "ms" << endl;
    MPI_Finalize();
    return 0;
}