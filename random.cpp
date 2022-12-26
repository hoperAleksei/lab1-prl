#include <iostream>
#include <thread>
#include <omp.h>
#include <iomanip>
#include "thread_num.cpp"


void set_num_threads(unsigned T);

typedef struct experiment_result_ {
    double result;
    double time;
    double speedup;
} experiment_result;

experiment_result* run_experiments(unsigned* V, unsigned count, double (*accumulate)(unsigned*, unsigned, unsigned, unsigned), unsigned min, unsigned max) {
    unsigned P = (unsigned)std::thread::hardware_concurrency();
    experiment_result* results = (experiment_result*)malloc(P * sizeof(experiment_result));
    for (unsigned i = 0; i < P; ++i) {
        double t0 = omp_get_wtime();
        set_num_threads(i + 1);
        results[i].result = accumulate(V, count, min, max);
        results[i].time = omp_get_wtime() - t0;
        results[i].speedup = results[0].time/results[i].time;
    }
    return results;
}

void print_experiment_results(const experiment_result* results){
    unsigned w = 10;
    std::cout << std::setw(w) << "T" << "\t"
    << std::setw(w) << "Time" << "\t"
    << std::setw(w) << "Result" << "\t"
    << std::setw(w) << "Speedup\n";
    for (unsigned T = 1; T <= std::thread::hardware_concurrency(); T++)
        std::cout << std::setw(w) << T << "\t"
        << std::setw(w) << results[T-1].time << "\t"
        << std::setw(w) << results[T-1].result<< "\t"
        << std::setw(w) << results[0].time/results[T-1].time << "\n";
}

void run_experiments_for(unsigned* V, unsigned count, double (*accumulate)(unsigned*, unsigned, unsigned, unsigned), unsigned min, unsigned max) {
    experiment_result* results = run_experiments(V, count, accumulate, min, max);
    print_experiment_results(results);
    free(results);
}

#include <vector>
#include <thread>

#define A 134775813
#define B 1
#define C 4294967296

unsigned get_num_threads();

std::vector<unsigned> pow_A(unsigned T) {
    std::vector<unsigned> result;
	result.reserve(T);
    result.emplace_back(A);
    for (unsigned i = 1; i < T + 1; i++) {
		unsigned next_A = (result[i-1] * A) % C;
        result.emplace_back(next_A);
    }
    return result;
}

#ifdef _MSC_VER
constexpr std::size_t CACHE_LINE = std::hardware_destructive_interference_size;
#else
#define CACHE_LINE 64
#endif

typedef struct element_t_
{
	alignas(CACHE_LINE) double value;
} element_t;

double randomize(unsigned* V, unsigned N, unsigned min, unsigned max) {

    unsigned T = get_num_threads();
    std::vector<unsigned> multipliers = pow_A(T);
    double sum = 0;
    std::vector<element_t> partial(T, element_t{0.0});
    std::vector<std::thread> threads;
    unsigned seed = std::time(0);
    for (std::size_t t = 0; t < T; ++t)
        threads.emplace_back([t, T, V, N, seed, &partial, multipliers, min, max]() {
        auto At = multipliers.back();
        unsigned off = (B * (At - 1) / (A - 1)) % C;
            unsigned x = ((seed * multipliers[t]) % C + (B % C * (multipliers[t] - 1) / (A - 1)) % C) % C;
            double acc = 0;
            for (size_t i = t; i < N; i += T) {
                V[i] = x % (max - min) + min;
                acc += V[i];
                x = (x * At) % C + off % C;
            }
            partial[t].value = acc;
            });
    for (auto& thread:threads)
        thread.join();
    for (unsigned i = 0; i < T; ++i)
        sum += partial[i].value;
    return sum / N;
}



int main() {
    unsigned T = get_num_threads();
    int N = 10000000;
    unsigned* V = new unsigned[N];
    run_experiments_for(V, N, randomize, 1, 99);
    return 0;
}