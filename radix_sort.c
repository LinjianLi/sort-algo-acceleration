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
  log_file = fopen("program-log-radix-sort.txt", "w");

  fprintf(log_file, "Number of elements: %d\nSeed: %f\n----------\n", n, seed);

  int num_threads[6] = {1,2,4,8,16,25};
  for (int i=0; i<6; ++i) {
    for (size_t radix_size = 7; radix_size <= 9; ++radix_size) {
      for (int buffer_size = 8; buffer_size <=32; buffer_size<<=1) {
        // To record the start and end time of the multiplication.
        struct timeval start;
        struct timeval end;
        unsigned long diff;
        gettimeofday(&start, NULL);

        RadixSortLSD_Buffer_OMP(arr, n, radix_size, buffer_size, num_threads[i]);

        fprintf(log_file,
                "D:%d\tK:%d\tT:%d\n"
                "Numbers at position 0.25, 0.5, 0.75: %d, %d, %d\n",
                radix_size, buffer_size, num_threads[i], arr[(size_t )(n*0.25)], arr[n>>1], arr[(size_t )(n*0.75)]);


        // Calculate the time spent by the multiplication.
        gettimeofday(&end,NULL);
        diff = (end.tv_sec-start.tv_sec)*1.0E6 + (end.tv_usec-start.tv_usec);
        setlocale(LC_NUMERIC, "");
        fprintf(log_file, "The time spent is %'ld microseconds\n----------\n", diff);
        fflush(log_file);
        // diff is time spent by the program, and the unit is microsecond

        sort_gen(arr, n, seed);   // Re-generate the random sequence.


      }

    }
  }

  fprintf(log_file, "Finished\n");
  fclose(log_file);
  free(arr);

  return 0;

}



