#include <iostream>
#include <windows.h>
#include <stdlib.h>

using namespace std;

const long N = 1048576; // matrix size

double A[N], B[N], res[N];

void init(int n)
{
    for (int i = 0; i < N; i++)
    {
        A[i] = i%10000;
    }
}
void  recursion(long n,long & sum)
{
    if (n == 1)
    {
        sum=A[0];
        return;
    }
    else
    {
        for (int i = 0; i < n / 2; i++)
            A[i] += A[n - i - 1];
        n = n / 2;
        recursion(n,sum);
    }
}
int main()
{
    for(long n=N; n>=32768; n/=2)
    {
        cout<<"规模为:"<<n<<endl;
        long long head, tail, freq;long sum = 0,sum1 = 0,sum2 = 0,sum3=0;
        init(N);
        for(int i=0; i<5; i++)
        {
            QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
            QueryPerformanceCounter((LARGE_INTEGER*)&head);
            for(int i=0; i<5; i++)
            {
                for (int i = 0; i < n; i++)
                    sum += A[i];
            }
        }
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        cout << "take:" << (tail - head) * 1000.0 / freq/5
             << "ms" << endl;

        init(N);
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        for(int i=0; i<5; i++)
        {
            for (int i = 0; i < n; i += 2)
            {
                sum1 += A[i];
                sum2 += A[i + 1];
            }
        }
        sum = sum1 + sum2;
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        cout << "take:" << (tail - head) * 1000.0 / freq /5<< "ms" << endl;

        init(N);
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        for(int i=0; i<5; i++)
        {
            recursion(n,sum3);
        }
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        cout << "take:" << (tail - head) * 1000.0 / freq/5
             << "ms" << endl;

        init(N);
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        for(int i=0; i<5; i++)
        {
            for (int m = n; m > 1; m /= 2) // log(n)个步骤
                for (int i = 0; i < m / 2; i++)
                    A[i] = A[i * 2] + A[i * 2 + 1];
        }
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        cout << "take:" << (tail - head) * 1000.0 / freq/5<< "ms" << endl;
        cout<<endl<<endl;
    }
}
