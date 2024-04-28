#include <iostream>
#include <windows.h>
#include <stdlib.h>

using namespace std;

const int N = 10000;

double A[N][N], B[N], res[N];

void init(int n)
{
    for (int i = 0; i < N; i++)
    {
        B[i] = i;
        for (int j = 0; j < N; j++)
            A[i][j] = i + j;
    }
}
int main()
{
    for(int n=N; n>=0; n-=2000)
    {
        cout<<n<<""<<endl;
        long long head, tail, freq;
        init(N);
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        for(int i=0; i<5; i++)
        {

            for (int i = 0; i < n; i++)
                res[i] = 0.0;
            for (int j = 0; j < n; j++)
                for (int i = 0; i < n; i++)
                    res[i] += A[j][i] * B[j];
        }
        // end time
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        cout << "take:" << (tail - head) * 1000.0 / freq / 5
             << "ms" << endl;

        init(N);
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        for(int i=0; i<5; i++)
        {

            for (int i = 0; i < n; i++)
                res[i] = 0.0;
            for (int j = 0; j < n; j++)
                for (int i = 0; i < n; i+=4)
                {
                    res[i] += A[j][i] * B[j];
                    res[i+1] += A[j][i+1] * B[j];
                    res[i+2] += A[j][i+2] * B[j];
                    res[i+3] += A[j][i+3] * B[j];
                }
        }
        // end time
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        cout << "take:" << (tail - head) * 1000.0 / freq / 5
             << "ms" << endl;
        cout<<endl<<endl;
    }
}
