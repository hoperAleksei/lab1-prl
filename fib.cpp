#include <omp.h>
#include <time.h>
#include <iostream>
#include <future>
#include "thread_num.cpp"


/*
 *Последовательная реализация вычисления суммы всех чисел ряда Фибоначчи. Обычная рекурсия
*/
unsigned fib_seq(unsigned n)
{
	if (n < 2)
		return n;

	return fib_seq(n - 1) + fib_seq(n - 2);
}

/*
 * Параллельная реализация вычисления суммы всех чисел ряда Фибоначчи.
 *
 * omp task - директива, которая ставит выполнение следующего блока кода - "задания",
 * в очередь выполнения. Если одно из ядер в данный момент свободно, то "задание"
 * из очереди на выполнение начинает выполняться. Таким образом, рекурсивно мы создаём
 * очередность выполнения заданий для выполнения на всех блоках процессора.
 *
 * Из-за особенностей реализации переменных суммации мы используем две разные
 * переменные-накопители и для каждой из них определяем собственный вид задания.
 *
 * Директива taskwait является своего рода барьером, который ожидает выполнения
 * всех запланированных заданий внутри каждого из рекурсивных блоков кода,
 * чтобы возвращаемое значение-результат операции не было неполным/побитым.
*/
unsigned fib_omp(unsigned n)
{
	if (n < 2)
		return n;

	unsigned r1, r2;
#pragma omp task shared(r1)
	{
		r1 = fib_omp(n - 1);
	}
#pragma omp task shared(r2)
	{
		r2 = fib_omp(n - 2);
	}
#pragma omp taskwait
	return r1 + r2;
}

/*
 *Параллельная реализация вычисления суммы всех чисел ряда Фибоначчи с помощью директивы async.
 * Сама по себе реализация один в один реализация с помощью task на omp.
 * Отличие в использовании библиотеки async.
 *
 * По сути своей создаем асинхронный запрос (async),
 * -> который должен выполняться на нескольких ядрах (std::launch::async),
 * -> вызывающий функцию fib_async_proc,
 * -> передающий ей значение предыдущего элемента ряда Фибоначчи (n-1) или позапредыдущего (n-2).
 *
 * Так как функция <future>.get() в return сама по себе синхронизированная при запросе результата,
 * то такой вызов ожидает выполнения всех операций, аки барьер. Оттого нам не нужно задумываться о
 * целостности данных.
*/
unsigned fib_async_proc(unsigned n) {
	{
		if (n < 2)
			return n;

		std::future<unsigned> res_l = std::async(std::launch::async, fib_async_proc, n-1);
		std::future<unsigned> res_r = std::async(std::launch::async, fib_async_proc, n-2);

		return res_l.get() + res_r.get();

	}
}

unsigned fib_async(unsigned n)
{
	return fib_async_proc(n);
}




//Стоит иметь ввиду - асинхронное выполнение ДОЛЖНО быть медленнее, чем последовательное,
int main(int argc, char *argv[]) {
	//При n > 20 оно должно встать колом
	int n = 20;

	double t0;
	unsigned res = 0;

	t0 = omp_get_wtime();
	res = fib_seq(n);
	std::cout << "seq:: " << res << " with time " << omp_get_wtime() - t0 << std::endl;


	t0 = omp_get_wtime();
	res = fib_omp(n);
	std::cout << "task: " << res << " with time " << omp_get_wtime() - t0 << std::endl;


	t0 = omp_get_wtime();
	res = fib_async(n);
	std::cout << "async: " << res << " with time " << omp_get_wtime() - t0 << std::endl;

}