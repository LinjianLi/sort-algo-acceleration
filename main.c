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
  int num_threads;
  if (n%25==0) {
    num_threads = 25;
  } else if (n%16==0) {
    num_threads = 16;
  } else if (n%8==0) {
    num_threads = 8;
  } else if (n%4==0) {
    num_threads = 4;
  } else if (n%2==0) {
    num_threads = 2;
  } else {
    num_threads = 1;
  }
  RadixSortLSD_Buffer_OMP(arr, n, 8, 16, num_threads);
  printf("The middle number: %d", arr[n>>1]);
  return 0;
}
