#include <iostream>
#include <chrono>
#include <omp.h>
#include <cstring>
#include <vector>
#include <thread>
#include <algorithm>
#include <memory>
#include <mutex>

//Динамический параллелизм (паралелизм на задачах, task_based parallelism)
//# pragma  omp parallel for reduction(+: sum) schedule(dinamic)

#define CACHE_LINE 64U
int N = 100000000;

struct partial_sum_T {
    double value [CACHE_LINE / sizeof(double)];
};


//double average_stat(const double * V, size_t n) {
//    double  s = 0;
//#pragma omp parallel for shedule(static)
//    for(size_t i = 0; i < n; ++i) {
//        s += V[i];
//    }
//
//    return s / n;
//}


#include <chrono>
#include <thread>
#include <vector>
#include <iostream>
#include <omp.h>


static unsigned threadsNum = std::thread::hardware_concurrency();
struct result_t {
    double value, milliseconds;
};


union partial_sum_t {
    double value;
    alignas(double) char pd[64];
};


void setThreadsNum(unsigned T) {
    threadsNum = T;
    omp_set_num_threads(T);
}

unsigned getThreadsNum() {
    return threadsNum;
}


void fillVector(double *v, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        v[i] = 1.0;
    }
}



//double average_par_1(const double *v, size_t n) {
//    unsigned T;
//    partial_sum_t *sums;
//    double result = 0;
//#pragma omp parallel shared(T, sums)
//    {
//        unsigned t = omp_get_thread_num();
//        double local_sum;
//#pragma omp single
//        {
//            T = (unsigned) omp_get_num_threads();
//            sums = (partial_sum_t *) malloc(T * sizeof(partial_sum_t));
//        }
//        for (size_t i = t; i < n; i += T) {
//            local_sum += v[i];
//        }
//        sums[t].value = local_sum;
//    }
//
//    for (size_t i = 0; i < T; ++i) {
//        result += sums[i].value;
//    }
//
//    free(sums);
//    return result / n;
//}
//
//
//double average_par_2(const double *v, size_t n) {
//    unsigned T;
//    partial_sum_t *sums;
//    double result = 0;
//#pragma omp parallel shared(T, sums)
//    {
//        unsigned t = omp_get_thread_num();
//        double local_sum;
//#pragma omp single
//        {
//            T = (unsigned) omp_get_num_threads();
//            sums = (partial_sum_t *) malloc(T * sizeof(partial_sum_t));
//        }
//
//        size_t n_t, i_0;
//
//        if (t < n % T) {
//            n_t = n / T + 1;
//            i_0 = n_t * t;
//        } else {
//            n_t = n / T;
//            i_0 = t * (n / T) + (n % T);
//        }
//
//        for (size_t i = i_0; i < n_t; ++i) {
//            local_sum = v[i];
//        }
//        sums[t].value = local_sum;
//    }
//
//    for (size_t i = 0; i < T; ++i) {
//        result += sums[i].value;
//    }
//
//    free(sums);
//    return result;
//}
//
//

result_t
run_experiment(double (*average)(const double *, size_t), const double *v, size_t n) {
    auto tm1 = std::chrono::steady_clock::now();
    double value = average(v, n);
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - tm1).count();
    result_t res{value, (double) time};
    return res;
}


void measure_scalability(auto averageFunction) {
    auto P = omp_get_num_procs();
    auto partial_res = std::make_unique<result_t[]>(P);
    auto v = std::make_unique<double[]>(N);
    fillVector(v.get(), N);

    std::cout << "num t | time | value | boost" << std::endl;

    for (auto T = 1; T <= P; ++T) {
        setThreadsNum(T);
        partial_res[T - 1] = run_experiment(averageFunction, v.get(), N);

        std::cout << T;
        std::cout << "\t" << partial_res[T - 1].milliseconds;
        std::cout << "\t" << partial_res[T - 1].value;
        std::cout << "\t" << partial_res[0].milliseconds / partial_res[T - 1].milliseconds;
        std::cout << std::endl;
    }
}


double average_par_static(const double *v, size_t n) {
    double sum = 0;
#pragma omp parallel for  reduction(+: sum) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        sum += v[i];
    }

    return sum / (double) n;
}


double average_par_dynamic(const double *v, size_t n) {
    double sum = 0;
#pragma omp parallel for reduction(+: sum) schedule(dynamic)
    for (size_t i = 0; i < n; ++i) {
        sum += v[i];
    }

    return sum / (double) n;
}

// ---------------------------------------------------------------------------------------------------------------------
// Фибоначи

unsigned  fib(unsigned n) {
    if (n < 2) return n;

    unsigned fibn1, fibn2;

#pragma omp task shared(fibn1)
{
    fibn1 = fib(n - 1);
}

#pragma omp task shared(fibn2)
{
    fibn2 = fib(n - 2);
}

#pragma omp taskwait
    return fibn1 + fibn2;
}

#include <future>


unsigned fib_async(unsigned n) {
    if (n < 2) return n;

    auto fibn1 = std::async([n] () {return fib_async(n-1);});
    auto fibn2 = std::async([] (unsigned n) {return fib_async(n-2);}, n);

    return fibn1.get() + fibn2.get();
}


int main() {

    std::cout << "fib" << std::endl;
    clock_t start = clock();
    std::cout << fib(20) << std::endl;
    clock_t end = clock();
    double seconds = (double)(end - start) / CLOCKS_PER_SEC;
    printf("The time: %f seconds\n", seconds);
    std::cout << std::endl;

    std::cout << "fib_async" << std::endl;
    clock_t start_async = clock();
    std::cout << fib_async(20) << std::endl;
    clock_t end_async = clock();
    double seconds_async = (double)(end_async - start_async) / CLOCKS_PER_SEC;
    printf("The time: %f seconds\n", seconds_async);
    std::cout << std::endl;

    std::cout << "AverageStatic:" << std::endl;
    measure_scalability(average_par_static);
    std::cout << std::endl;

    std::cout << "AverageDynamic:" << std::endl;
    measure_scalability(average_par_dynamic);

}