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
void  recursion2(long n,long & sum)
{
    if (n == 1)
    {
        sum=A[0];
        return;
    }
    else
    {
        for (int i = 0; i < n / 2; i+=2)
            {A[i] += A[n - i - 1];
            A[i+1] += A[n - i - 2];
            }

        n = n / 2;
        recursion(n,sum);
    }
}
int main()
{
    int i=0;
    for(long n=N/8;i<4; n*=2,i++)
    {
        cout<<"¹æÄ£Îª:"<<n<<endl;
        long long head, tail, freq;long sum = 0,sum3=0;

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
            recursion2(n,sum);
        }
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        cout << "take:" << (tail - head) * 1000.0 / freq/5
             << "ms" << endl;
        cout<<endl<<endl;
    }
}
