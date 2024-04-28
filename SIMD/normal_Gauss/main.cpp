#include <iostream>
#include <windows.h>
#include <pmmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <tmmintrin.h>
#include <nmmintrin.h>
#include <immintrin.h>
using namespace std;

const int N = 2000;

float matrix[N][N];
float matrix2[N][N];
float matrix3[N][N];
void init()
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            matrix3[i][j] = 0;
        }
        matrix3[i][i] = 1.0;
        for (int j = i + 1; j < N; j++)
            matrix3[i][j] = rand();
    }

    for (int k = 0; k < N; k++)
    {
        for (int i = k + 1; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                matrix3[i][j] += matrix3[k][j];
            }
        }
    }
}

void set()
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            matrix[i][j] = matrix3[i][j];
            matrix2[i][j] = matrix3[i][j];
        }
    }
}

void normal()
{
    for (int k = 0; k < N; k++)
    {
        for (int i = k + 1; i < N; i++)
            matrix[k][i] = matrix[k][i] / matrix[k][k];
        matrix[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            for (int j = k + 1; j < N; j++)
                matrix[i][j] -= matrix[i][k] * matrix[k][j];
            matrix[i][k] = 0;
        }
    }
}


void normal_cache()
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < i; j++)
        {
            matrix2[j][i] = matrix[i][j];
            matrix[i][j] = 0;
        }
    }
    for (int k = 0; k < N; k++)
    {
        for (int i = k + 1; i < N; i++)
            matrix[k][i] = matrix[k][i] / matrix[k][k];
        matrix[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            for (int j = k + 1; j < N; j++)
                matrix[i][j] = matrix[i][j] - matrix2[k][i] * matrix[k][j];
            matrix[i][k] = 0;
        }
    }
}
void normal_cache2()
{
    for (int k = 0; k < N; k++)
    {
        for (int i = k + 1; i < N; i++)
            matrix[k][i] = matrix[k][i] / matrix[k][k];
        matrix[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            for (int j = k + 1; j < N; j++)
                matrix[i][j] = matrix[i][j] - matrix2[k][i] * matrix[k][j];
            matrix[i][k] = 0;
        }
    }
}

void sse_mul()
{
    __m128 t1, t2, t3;
    for (int k = 0; k < N; k++)
    {
        for (int j = k + 1; j < N; j++)
            matrix[k][j] = matrix[k][j] / matrix[k][k];
        matrix[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            t1 = _mm_set_ps(matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k]);
            int j = k + 1;
            for (j; j < N - 3; j += 4)
            {
                t2 = _mm_loadu_ps(matrix[i] + j);
                t3 = _mm_loadu_ps(matrix[k] + j);
                t3 = _mm_mul_ps(t1, t3);
                t2 = _mm_sub_ps(t2, t3);
                _mm_storeu_ps(matrix[i] + j, t2);
            }
            for (j; j < N; j++)
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            matrix[i][k] = 0;
        }
    }
}
void sse_mul_aligned()
{
    __m128 t1, t2, t3;
    for (int k = 0; k < N; k++)
    {
        for (int j = k + 1; j < N; j++)
            matrix[k][j] = matrix[k][j]* 1.0 / matrix[k][k];
        matrix[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            float temp[4] = {matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k]};
            t1 = _mm_load_ps(temp);
            int j = k + 1;
            for (j; j % 4 != 0; j++)
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            for (j; j < N - 3; j += 4)
            {
                t2 = _mm_load_ps(matrix[i] + j);
                t3 = _mm_load_ps(matrix[k] + j);
                t3 = _mm_mul_ps(t1, t3);
                t2 = _mm_sub_ps(t2, t3);
                _mm_store_ps(matrix[i] + j, t2);
            }
            for (j; j < N; j++)
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            matrix[i][k] = 0;
        }
    }
}

void sse_div()
{
    __m128 t1, t2, t3;
    for (int k = 0; k < N; k++)
    {
        t1 = _mm_set_ps(matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k]);
        int j = k + 1;
        for (j; j < N - 3; j += 4)
        {
            t2 = _mm_loadu_ps(matrix[k] + j);
            t3 = _mm_div_ps(t2, t1);
            _mm_storeu_ps(matrix[k] + j, t3);
        }
        for (j; j < N; j++)
            matrix[k][j] = matrix[k][j] / matrix[k][k];
        matrix[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            for (int j = k + 1; j < N; j++)
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            matrix[i][k] = 0;
        }
    }
}
void sse_div_aligned()
{
    __m128 t1, t2, t3;
    for (int k = 0; k < N; k++)
    {
        t1 = _mm_set_ps(matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k]);
        int j = k + 1;
        for (j; j % 4 != 0; j++)
            matrix[k][j] = matrix[k][j] / matrix[k][k];
        for (j; j < N - 3; j += 4)
        {
            t2 = _mm_load_ps(matrix[k] + j);
            t3 = _mm_div_ps(t2, t1);
            _mm_store_ps(matrix[k] + j, t3);
        }
        for (j; j < N; j++)
            matrix[k][j] = matrix[k][j] / matrix[k][k];
        matrix[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            for (int j = k + 1; j < N; j++)
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            matrix[i][k] = 0;
        }
    }
}
void sse()
{
    __m128 t1, t2, t3;
    for (int k = 0; k < N; k++)
    {
        t1 = _mm_set_ps(matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k]);
        int j = k + 1;
        for (j; j < N - 3; j += 4)
        {
            t2 = _mm_loadu_ps(matrix[k] + j);
            t3 = _mm_div_ps(t2, t1);
            _mm_storeu_ps(matrix[k] + j, t3);
        }
        for (j; j < N; j++)
            matrix[k][j] = matrix[k][j] / matrix[k][k];
        matrix[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            float temp[4] = {matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k]};
            t1 = _mm_loadu_ps(temp);
            int j = k + 1;
            for (j; j < N - 3; j += 4)
            {
                t2 = _mm_loadu_ps(matrix[i] + j);
                t3 = _mm_loadu_ps(matrix[k] + j);
                t3 = _mm_mul_ps(t1, t3);
                t2 = _mm_sub_ps(t2, t3);
                _mm_storeu_ps(matrix[i] + j, t2);
            }
            for (j; j < N; j++)
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            matrix[i][k] = 0;
        }
    }
}
void sse_aligned()
{
    __m128 t1, t2, t3;
    for (int k = 0; k < N; k++)
    {
        t1 = _mm_set_ps(matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k]);
        int j = k + 1;
        for (j; j % 4 != 0; j++)
            matrix[k][j] = matrix[k][j] / matrix[k][k];
        for (j; j < N - 3; j += 4)
        {
            t2 = _mm_load_ps(matrix[k] + j);
            t3 = _mm_div_ps(t2, t1);
            _mm_store_ps(matrix[k] + j, t3);
        }
        for (j; j < N; j++)
            matrix[k][j] = matrix[k][j] / matrix[k][k];
        matrix[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            float temp[4] = {matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k]};
            t1 = _mm_loadu_ps(temp);
            int j = k + 1;
            for (j; j % 4 != 0; j++)
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            for (j; j < N - 3; j += 4)
            {
                t2 = _mm_load_ps(matrix[i] + j);
                t3 = _mm_load_ps(matrix[k] + j);
                t3 = _mm_mul_ps(t1, t3);
                t2 = _mm_sub_ps(t2, t3);
                _mm_store_ps(matrix[i] + j, t2);
            }
            for (j; j < N; j++)
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            matrix[i][k] = 0;
        }
    }
}
void avx()
{
    __m256 t1, t2, t3;
    for (int k = 0; k < N; k++)
    {
        float temp1[8] = {matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k]};
        t1 = _mm256_loadu_ps(temp1);
        int j = k + 1;
        for (j; j < N - 7; j += 8)
        {
            t2 = _mm256_loadu_ps(matrix[k] + j);
            t3 = _mm256_div_ps(t2, t1);
            _mm256_storeu_ps(matrix[k] + j, t3);
        }
        for (j; j < N; j++)
            matrix[k][j] = matrix[k][j] / matrix[k][k];
        matrix[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            float temp2[8] = {matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k]};
            t1 = _mm256_loadu_ps(temp2);
            j = k + 1;
            for (j; j < N - 7; j += 8)
            {
                t2 = _mm256_loadu_ps(matrix[i] + j);
                t3 = _mm256_loadu_ps(matrix[k] + j);
                t3 = _mm256_mul_ps(t1, t3);
                t2 = _mm256_sub_ps(t2, t3);
                _mm256_storeu_ps(matrix[i] + j, t2);
            }
            for (j; j < N; j++)
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            matrix[i][k] = 0;
        }
    }
}
void _aligned()
{
    __m256 t1, t2, t3;
    for (int k = 0; k < N; k++)
    {
        float temp1[8] = {matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k], matrix[k][k]};
        t1 = _mm256_loadu_ps(temp1);
        int j = k + 1;
        for (j; j % 8 != 0; j++)
            matrix[k][j] = matrix[k][j] / matrix[k][k];
        for (j; j < N - 7; j += 8)
        {
            t2 = _mm256_loadu_ps(matrix[k] + j);
            t3 = _mm256_div_ps(t2, t1);
            _mm256_storeu_ps(matrix[k] + j, t3);
        }
        for (j; j < N; j++)
            matrix[k][j] = matrix[k][j] / matrix[k][k];
        matrix[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            float temp2[8] = {matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k], matrix[i][k]};
            t1 = _mm256_loadu_ps(temp2);
            j = k + 1;
            for (j; j % 8 != 0; j++)
            matrix[k][j] = matrix[k][j] / matrix[k][k];
            for (j; j < N - 7; j += 8)
            {
                t2 = _mm256_loadu_ps(matrix[i] + j);
                t3 = _mm256_loadu_ps(matrix[k] + j);
                t3 = _mm256_mul_ps(t1, t3);
                t2 = _mm256_sub_ps(t2, t3);
                _mm256_storeu_ps(matrix[i] + j, t2);
            }
            for (j; j < N; j++)
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            matrix[i][k] = 0;
        }
    }
}

int main()
{
    int epoch = 10;
    long long head, tail, freq; // timers
    long double timer;
    init();

    timer = 0;
    for (int i = 0; i < epoch; i++)
    {
        set();
        QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
        QueryPerformanceCounter((LARGE_INTEGER *)&head);
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < i; j++)
            {
                matrix2[j][i] = matrix[i][j];
                matrix[i][j] = 0;
            }
        }
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        timer += (tail - head) * 1000.0 / freq;
    }
    cout << "qiuni cost: " << timer / epoch << "ms" << endl;

    timer = 0;
    for (int i = 0; i < epoch; i++)
    {
        set();
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < i; j++)
            {
                matrix2[j][i] = matrix[i][j];
                matrix[i][j] = 0;
            }
        }
        QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
        QueryPerformanceCounter((LARGE_INTEGER *)&head);
        normal_cache2();
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        timer += (tail - head) * 1000.0 / freq;
    }
    cout << "normal_cache2 cost: " << timer / epoch << "ms" << endl;

    timer = 0;
    for (int i = 0; i < epoch; i++)
    {
        set();
        QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
        QueryPerformanceCounter((LARGE_INTEGER *)&head);
        normal();
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        timer += (tail - head) * 1000.0 / freq;
    }
    cout << "normal cost: " << timer / epoch << "ms" << endl;


    timer = 0;
    for (int i = 0; i < epoch; i++)
    {
        set();
        QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
        QueryPerformanceCounter((LARGE_INTEGER *)&head);
        normal_cache();
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        timer += (tail - head) * 1000.0 / freq;
    }
    cout << "normal_cache cost: " << timer / epoch << "ms" << endl;


    timer = 0;
    for (int i = 0; i < epoch; i++)
    {
        set();
        QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
        QueryPerformanceCounter((LARGE_INTEGER *)&head);
        sse_mul();
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        timer += (tail - head) * 1000.0 / freq;
    }
    cout << "sse_mul cost: " << timer / epoch << "ms" << endl;


    timer = 0;
    for (int i = 0; i < epoch; i++)
    {
        set();
        QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
        QueryPerformanceCounter((LARGE_INTEGER *)&head);
        sse_mul_aligned();
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        timer += (tail - head) * 1000.0 / freq;
    }
    cout << "sse_mul_aligned cost: " << timer / epoch << "ms" << endl;


    timer = 0;
    for (int i = 0; i < epoch; i++)
    {
        set();
        QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
        QueryPerformanceCounter((LARGE_INTEGER *)&head);
        sse_div();
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        timer += (tail - head) * 1000.0 / freq;
    }
    cout << "sse_div cost: " << timer / epoch << "ms" << endl;


    timer = 0;
    for (int i = 0; i < epoch; i++)
    {
        set();
        QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
        QueryPerformanceCounter((LARGE_INTEGER *)&head);
        sse_div_aligned();
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        timer += (tail - head) * 1000.0 / freq;
    }
    cout << "sse_div_aligned cost: " << timer / epoch << "ms" << endl;


    timer = 0;
    for (int i = 0; i < epoch; i++)
    {
        set();
        QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
        QueryPerformanceCounter((LARGE_INTEGER *)&head);
        sse();
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        timer += (tail - head) * 1000.0 / freq;
    }
    cout << "sse cost: " << timer / epoch << "ms" << endl;


    timer = 0;
    for (int i = 0; i < epoch; i++)
    {
        set();
        QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
        QueryPerformanceCounter((LARGE_INTEGER *)&head);
        sse_aligned();
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        timer += (tail - head) * 1000.0 / freq;
    }
    cout << "sse_aligned cost: " << timer / epoch << "ms" << endl;


    timer = 0;
    for (int i = 0; i < epoch; i++)
    {
        set();
        QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
        QueryPerformanceCounter((LARGE_INTEGER *)&head);
        avx_lu();
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        timer += (tail - head) * 1000.0 / freq;
    }
    cout << "avx cost: " << timer / epoch << "ms" << endl;


    timer = 0;
    for (int i = 0; i < epoch; i++)
    {
        set();
        QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
        QueryPerformanceCounter((LARGE_INTEGER *)&head);
        avx();
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        timer += (tail - head) * 1000.0 / freq;
    }
    cout << "avx_aligned cost: " << timer / epoch << "ms" << endl;


    return 0;
}
