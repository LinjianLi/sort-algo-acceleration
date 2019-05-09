#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <locale.h>
#include "GenRandSequence.h"
#include "SortAlgoImpleAccel.h"

int main(int argc, char *argv[]) {
  int n = (int) strtol(argv[1], NULL, 10);
  float seed = strtof(argv[2], NULL);
  int *arr = malloc(n * sizeof(*arr));  // int *arr = new int[n];
  sort_gen(arr, n, seed);
  int num_threads = 24;
  RadixSortLSD_Buffer_OMP(arr, n, 8, 16, num_threads);
  printf("The middle number: %d", arr[n>>1]);
  return 0;
}
