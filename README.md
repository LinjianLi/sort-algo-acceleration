# sort-algo-acceleration

## Performance

CPU: XEON E2620, 2.00GHZ, 2 sockets, 6 cores per socket, 2 threads per core

OS： CentOS release 6.10

GCC: gcc 7.1.0

Sequence size: 1,000,000,000 int

Runtime:
* <cstdlib> qsort(): 457s
* OMP 24 threads: 1972s
* Radix Sort, 8 bits per radix, 16 elements buffer per radix, OMP 16 threads: 22s
* Merge Sort: buggy (68s)

