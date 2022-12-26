#include <iostream>
#include <omp.h>
#include <cmath>

using namespace std;

#define ull unsigned long long
#define A0 2
#define B 1
#define C 100
#define x_min 0
#define x_max 99
#define x0 0
#define n 60
#define T 4
ull acc = 0;
ull V[n];
ull A[n];

// | 1 | 3 | 7 | 15 | 31 | 63 | 27 | n=7

void srand(int seed) {
    omp_set_num_threads(n);
    ull x, xnew;
    A[0] = A0;

    for (int i = 1; i < n; i++) A[i] = A0 * A[i - 1];

#pragma omp parallel shared(V, A, acc)
    {
        int t = omp_get_thread_num();
        #pragma omp for
        for (int i = 0; i < n; i++) {
            x = (A[t] * seed + (A[t] - 1) / (A[0] - 1) * B) % C;
            xnew = x * ((x_max - x_min + 1) / C) + x_min;
            V[t] = xnew;
            acc += xnew;
        }
    }


//    #pragma omp parallel for
//    for (int i = 0; i < n; i++) {
//        int x = (A[i] * seed + (A[i] - 1) / (A[0] - 1) * B) % C;
//        int xnew = x * ((x_max - x_min + 1) / C) + x_min;
//        V[t] = xnew;
//        acc += xnew;
//        t += 1;
//    }
}

int rand() {
    srand(0);
    return acc / n;
}

int main() {
    cout << rand() << '\n';

    for (ull & i: V) cout << i << ' ';

    return 0;
}
