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

  FILE *log_file, *runtime_control_file;
  log_file = fopen("program-log.txt", "w");
  runtime_control_file = fopen("runtime-control.txt", "r");

  fprintf(log_file, "Number of elements: %d\nSeed: %f\n----------\n", n, seed);

  int s;
  //printf("Select a method: ");
  fscanf(runtime_control_file, "%d", &s);

  size_t radix_size = 8;

  while (s>0) {


    // To record the start and end time of the multiplication.
    struct timeval start;
    struct timeval end;
    unsigned long diff;
    gettimeofday(&start, NULL);


    switch (s) {
      case 1:
        qsort(arr, n, sizeof(int), compare);
        break;
      case 2:
        RadixSortLSD(arr,n,radix_size);
        break;
      case 3:
        RadixSortLSD_Buffer_SingleThread(arr, n, radix_size, 16);
        break;
      case 4:
        RadixSortLSD_Buffer_SingleThread(arr, n, radix_size, 32);
        break;
      case 5:
        RadixSortLSD_Buffer_SingleThread(arr, n, radix_size, 64);
        break;
      case 6:
        RadixSortLSD_Buffer_SingleThread(arr, n, radix_size, 128);
        break;
      case 7:
        RadixSortLSD_MultiThreads(arr, n, radix_size, 2);
        break;
      case 8:
        RadixSortLSD_MultiThreads(arr, n, radix_size, 4);
        break;
      case 9:
        RadixSortLSD_MultiThreads(arr, n, radix_size, 8);
        break;
      case 10:
        RadixSortLSD_MultiThreads(arr, n, radix_size, 16);
        break;
      case 11:
        RadixSortLSD_Buffer_MultiThreads(arr, n, radix_size, 16, 4);
        break;
      case 12:
        RadixSortLSD_Buffer_MultiThreads(arr, n, radix_size, 32, 4);
        break;
      case 13:
        RadixSortLSD_Buffer_MultiThreads(arr, n, radix_size, 64, 4);
        break;
      case 14:
        RadixSortLSD_Buffer_MultiThreads(arr, n, radix_size, 128, 4);
        break;
      case 15:

        break;
      default:
        fprintf(log_file, "Invalid input!\n");
    }

    fprintf(log_file,
            "method %d\n"
            "Numbers at position 0.25, 0.5, 0.75: %d, %d, %d\n",
            s, arr[(size_t )(n*0.25)], arr[n>>1], arr[(size_t )(n*0.75)]);


    // Calculate the time spent by the multiplication.
    gettimeofday(&end,NULL);
    diff = 1000000*(end.tv_sec-start.tv_sec) + (end.tv_usec-start.tv_usec);
    setlocale(LC_NUMERIC, "");
    fprintf(log_file, "The time spent is %'ld microseconds\n----------\n", diff);
    fflush(log_file);
    // diff is time spent by the program, and the unit is microsecond


    sort_gen(arr, n, seed);   // Re-generate the random sequence.

    //printf("Select a method: ");
    fscanf(runtime_control_file, "%d", &s);
  }


  fclose(log_file);
  fclose(runtime_control_file);
  free(arr);

  return 0;
}
