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


  FILE *log_file;
  log_file = fopen("program-log-merge-sort.txt", "w");

  fprintf(log_file, "Number of elements: %d\nSeed: %f\n----------\n", n, seed);

  for (int num_threads=16; num_threads>=1; num_threads>>=1) {
        struct timeval start;
        struct timeval end;
        unsigned long diff;
        gettimeofday(&start, NULL);

        Merge_Radix_OMP(arr, n, 8, num_threads);

        fprintf(log_file,
                "T:%d\n"
                "Numbers at position 0.25, 0.5, 0.75: %d, %d, %d\n",
                 num_threads, arr[(size_t )(n*0.25)], arr[n>>1], arr[(size_t )(n*0.75)]);


        // Calculate the time spent by the multiplication.
        gettimeofday(&end,NULL);
        diff = (end.tv_sec-start.tv_sec)*1.0E6 + (end.tv_usec-start.tv_usec);
        setlocale(LC_NUMERIC, "");
        fprintf(log_file, "The time spent is %'ld microseconds\n----------\n", diff);
        fflush(log_file);
        // diff is time spent by the program, and the unit is microsecond

        sort_gen(arr, n, seed);   // Re-generate the random sequence.
  }

  fprintf(log_file, "Finished\n");
  fclose(log_file);
  free(arr);

  return 0;

}



