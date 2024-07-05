#include<iostream>
#include<fstream>
#include<sstream>
#include<cmath>
#include<bitset>
#include<cuda_runtime.h>

using namespace std;

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

__global__ void cuda_func(unsigned int* d_E, unsigned int* d_R, int* d_First, int E_RowNum, int E_ColNum, int R_ColNum)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < E_RowNum)
    {
        while (d_First[i] != -1)
        {
            bool isnull = true;
            for (int h = 0; h < R_ColNum; h++)
            {
                if (d_R[d_First[i] * R_ColNum + h] != 0)
                {
                    isnull = false;
                    for (int k = 0; k < E_ColNum; k++)
                    {
                        d_E[i * E_ColNum + k] ^= d_R[d_First[i] * R_ColNum + k];
                    }

                    int j, k;
                    for (k = 0; k < E_ColNum; k++)
                        if (d_E[i * E_ColNum + k] != 0)
                            break;

                    if (k == E_ColNum)
                    {
                        d_First[i] = -1;
                        break;
                    }
                    unsigned int temp = d_E[i * E_ColNum + k];
                    for (j = 0; temp != 0; j++)
                        temp >>= 1;

                    d_First[i] = E_ColNum * 32 - (k + 1) * 32 + j - 1;
                }
            }
            if (isnull)
            {
                for (int j = 0; j < R_ColNum; j++)
                    d_R[d_First[i] * R_ColNum + j] = d_E[i * E_ColNum + j];
                break;
            }
        }
    }
}

void cuda()
{
    unsigned int* d_E;
    unsigned int* d_R;
    int* d_First;

    size_t E_size = E_RowNum * E_ColNum * sizeof(unsigned int);
    size_t R_size = R_RowNum * R_ColNum * sizeof(unsigned int);
    size_t First_size = E_RowNum * sizeof(int);

    cudaMalloc(&d_E, E_size);
    cudaMalloc(&d_R, R_size);
    cudaMalloc(&d_First, First_size);

    cudaMemcpy(d_E, E, E_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_R, R, R_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_First, First, First_size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (E_RowNum + blockSize - 1) / blockSize;

    cuda_func<<<numBlocks, blockSize>>>(d_E, d_R, d_First, E_RowNum, E_ColNum, R_ColNum);

    cudaMemcpy(E, d_E, E_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(R, d_R, R_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(First, d_First, First_size, cudaMemcpyDeviceToHost);

    cudaFree(d_E);
    cudaFree(d_R);
    cudaFree(d_First);
}

int main()
{
    Init();
    Init_E();
    Init_R();
    cuda();
    return 0;
}
