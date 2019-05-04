//
// Created by LinjianLi on 2019/4/19.
//

#ifndef SORT_ALGO_ACCELERATION_SORTALGOIMPLEACCEL_H
#define SORT_ALGO_ACCELERATION_SORTALGOIMPLEACCEL_H

#include <memory.h>
#include <string.h>  // memset
#include <math.h>
#include <stdint.h>
#include "omp.h"


int compare (const void *a, const void *b){
  return ( *(int*)a - *(int*)b );
}

void PrintArray_Int(int *arr, size_t arr_size, size_t line_size) {
  printf("{ ");
  for (size_t i=0; i<arr_size-1; ++i) {
    printf("%d, ", arr[i]);
    if ((i+1)%line_size==0) {printf("\n");}
  }
  printf("%d }\n----------\n", arr[arr_size-1]);
}

void PrintArray_SizeT(size_t *arr, size_t arr_size, size_t line_size) {
  printf("{ ");
  for (size_t i=0; i<arr_size-1; ++i) {
    printf("%lu, ", arr[i]);
    if ((i+1)%line_size==0) {printf("\n");}
  }
  printf("%lu }\n", arr[arr_size-1]);
}


/**
 * @param array : elements must greater or equal to zero
 * @param bits_leftmost_pos
 * @param bits_rightmost_pos
 *      Assume using "LSB 0" numbering,
 *      which means that bits numbering start at 0 for the least significant bit.
 *      So, bits_leftmost_pos > bits_rightmost_pos.
 */
void CountSortOnBits(int *array, size_t array_size,
                     size_t bits_leftmost_pos, size_t bits_rightmost_pos) {


  const size_t radix_size = bits_leftmost_pos-bits_rightmost_pos+1;
  const uint64_t num_radixes = (uint64_t ) 1 << radix_size;

  const size_t num_bits_to_shift = bits_rightmost_pos;
  const uint32_t mask = 0xFFFFFFFF >> (32-radix_size);


  // Get specific radixes of each element in array.
  int *array_radixes = malloc(array_size * sizeof(*array_radixes));
  for (size_t i=0; i<array_size; ++i) {
    array_radixes[i] = (array[i] >> num_bits_to_shift) & mask;
  }


  // Initialize histogram to all zero.
  size_t *histogram = malloc(num_radixes * sizeof(*histogram));
  memset(histogram, 0, num_radixes*sizeof(*histogram));


  // Count and update the histogram.
  // The shifting operation is because we are doing counting on specific bits.
  for (size_t i=0; i<array_size; ++i) {
    ++histogram[ array_radixes[i] ];
  }


  // Calculate the prefix sum. The result is the ending offset of each radix.
  for (size_t i=1; i<num_radixes; ++i) {

    // This implementation is not good for parallel.
    histogram[i] += histogram[i-1];

    // This implementation is good for parallel.
//    for (int j=i-1; j>=0; --j) {
//      histogram[i] += histogram[j];
//    }

  }


  int *output_sorted = malloc(array_size * sizeof(*output_sorted));

  // Fill elements into the output array, according the histogram.
  // For the stable sorting algorithm, counting sort now need to
  // scan the original array from the end to the start.
  // If scanning from the start, the output is unstable.
  for (size_t i=1; i<=array_size; ++i) {
    int current_radix = array_radixes[array_size-i];
    output_sorted[--histogram[current_radix]] = array[array_size-i];
  }

  memcpy(array, output_sorted, array_size*sizeof(*array));
  free(output_sorted);
  free(array_radixes);
  free(histogram);
}


void CountSortOnBits_OMP(int *array, const size_t array_size,
                         const size_t bits_leftmost_pos, const size_t bits_rightmost_pos,
                         const size_t num_threads) {

  if ((array_size/num_threads)*num_threads != array_size) {
    fprintf(stderr,"For now, the algorithm only support the input that the array size is multiple of number of threads!\n");
    return;
  }

  omp_set_num_threads(num_threads);


  const size_t radix_size = bits_leftmost_pos-bits_rightmost_pos+1;
  const uint64_t num_radixes = (uint64_t ) 1 << radix_size;

  const size_t num_bits_to_shift = bits_rightmost_pos;
  const uint32_t mask = 0xFFFFFFFF >> (32-radix_size);

  const size_t num_elems_per_section = array_size/num_threads;

  int *output_sorted = malloc(array_size * sizeof(*output_sorted));

  // Get specific radixes of each element in array.
  int *array_radixes = malloc(array_size * sizeof(*array_radixes));
  #pragma omp parallel for
  for (size_t i=0; i<array_size; ++i) {
    array_radixes[i] = (array[i] >> num_bits_to_shift) & mask;
  }



  // Initialize local histograms to all zero.
  // Imagine this local histogram is a 2D array: [num_threads][num_radixes]
  // Each thread occupy one line.
  int *local_histograms = malloc(num_threads * num_radixes * sizeof(*local_histograms));
  memset(local_histograms, 0, num_threads*num_radixes*sizeof(*local_histograms));


  #pragma omp parallel
  {

    // Count and update the local histogram.
    // The shifting operation is because we are doing counting on specific bits.
    size_t current_thread_id = omp_get_thread_num();
    size_t array_section_start = current_thread_id * num_elems_per_section;
    size_t thread_histogram_start_pos = current_thread_id * num_radixes;
    for (size_t i = array_section_start; i < array_section_start + num_elems_per_section; ++i) {
      ++local_histograms[thread_histogram_start_pos + array_radixes[i]];
    }
  }

  // Calculate the prefix sum. The result is the ending offset of each radix.
  // According to the paper
  //   [Satish et al. - 2010 - Fast sort on CPUs and GPUs...]
  // But make some changes.
  // After calculating the prefix sum, the values of the histogram
  //   * indicate the STARTING write offset in the paper.
  //   * indicate the ENDING write offset in this code.
  int *updated_local_histograms = malloc(num_threads * num_radixes * sizeof(*updated_local_histograms));
  memset(updated_local_histograms, 0, num_threads*num_radixes*sizeof(*updated_local_histograms));
  #pragma omp parallel for
  for (size_t i=0; i<num_threads*num_radixes; ++i) {  // for each entry of histogram
    // calculate the prefix sum according to the formula in the paper and make some modification

    // Sum the number of elements whose radix are smaller than this.
    for (size_t j=0; j<num_threads; ++j) {
      for (size_t k=0; k<i%num_radixes; ++k) {
        updated_local_histograms[i] += local_histograms[j*num_radixes + k];
      }
    }

    // Sum the number of elements whose radix are the same as this
    // but were calculated by the threads whose ID are smaller than or equal to this.
    for (size_t j=0; j<=(size_t)(i/num_radixes); ++j) {
      updated_local_histograms[i] += local_histograms[j*num_radixes + i%num_radixes];
    }
  }

  memcpy(local_histograms, updated_local_histograms, num_threads * num_radixes * sizeof(*local_histograms));
  free(updated_local_histograms);

  // Fill elements into the output array, according the histogram.
  // For the stable sorting algorithm, counting sort now need to
  // scan the original array from the end to the start.
  // If scanning from the start, the output is unstable.
  #pragma omp parallel
  {
    size_t current_thread_id = omp_get_thread_num();
    size_t array_section_start = current_thread_id * num_elems_per_section;
    size_t thread_histogram_start_pos = current_thread_id * num_radixes;
    // Note the STABILITY of the sorting algorithm!!!
    for (int i = array_section_start+num_elems_per_section-1; i >= (int)array_section_start; --i) {
      int current_radix = array_radixes[i];
      int write_pointer = --local_histograms[thread_histogram_start_pos + current_radix];
      output_sorted[write_pointer] = array[i];
    }
  }

  memcpy(array, output_sorted, array_size*sizeof(*array));
  free(output_sorted);
  free(local_histograms);
  free(array_radixes);
}


/**
 * @param array : elements must greater or equal to zero
 * @param bits_leftmost_pos
 * @param bits_rightmost_pos
 *      Assume using "LSB 0" numbering,
 *      which means that bits numbering start at 0 for the least significant bit.
 *      So, bits_leftmost_pos > bits_rightmost_pos.
 */
void CountSortOnBits_Buffer_1Thread(int *array, size_t array_size,
                                    size_t bits_leftmost_pos, size_t bits_rightmost_pos,
                                    size_t num_elems_per_radix_buf) {


  const size_t radix_size = bits_leftmost_pos-bits_rightmost_pos+1;
  const uint32_t num_radixes = (uint32_t ) 1 << radix_size;

  const size_t num_bits_to_shift = bits_rightmost_pos;
  const uint32_t mask = 0xFFFFFFFF >> (32-radix_size);


  // Get specific radixes of each element in array.
  int *array_radixes = malloc(array_size * sizeof(*array_radixes));
  for (size_t i=0; i<array_size; ++i) {
    array_radixes[i] = (array[i] >> num_bits_to_shift) & mask;
  }

  // Initialize histogram to all zero.
  size_t *histogram = malloc(num_radixes * sizeof(*histogram));
  memset(histogram, 0, num_radixes*sizeof(*histogram));


  // Count and update the histogram.
  // The shifting operation is because we are doing counting on specific bits.
  for (size_t i=0; i<array_size; ++i) {
    ++histogram[ array_radixes[i] ];
  }


  // Calculate the prefix sum. The result is the ending offset of each radix.
  for (size_t i=1; i<num_radixes; ++i) {

    // This implementation is not good for parallel.
    histogram[i] += histogram[i-1];

    // This implementation is good for parallel.
//    for (int j=i-1; j>=0; --j) {
//      histogram[i] += histogram[j];
//    }

  }


  // Initialize buffers. And set write pointers to all zero.
  int *buffers = malloc(num_radixes * num_elems_per_radix_buf * sizeof(*buffers));
  size_t *buf_write_pointers = malloc(num_radixes * sizeof(*buf_write_pointers));
  memset(buf_write_pointers, 0, num_radixes*sizeof(*buf_write_pointers));

  int *output_sorted = malloc(array_size * sizeof(*output_sorted));

  // Fill elements into the output array, according the histogram.
  // For the stable sorting algorithm, counting sort now need to
  // scan the original array from the end to the start.
  // If scanning from the start, the output is unstable.
  for (size_t i=1; i<=array_size; ++i) {

    int current_radix = array_radixes[array_size-i];

    // Put element into buffer.
    buffers[current_radix * num_elems_per_radix_buf
            +
            (buf_write_pointers[current_radix]++)] = array[array_size-i];

    // If the buffer is full, flush it.
    if (buf_write_pointers[current_radix]==num_elems_per_radix_buf) {
      size_t buf_pos_this_radix = current_radix * num_elems_per_radix_buf;
      for (size_t j=0; j<num_elems_per_radix_buf; ++j) {  // Flush buffer to memory.
        output_sorted[--histogram[current_radix]] = buffers[buf_pos_this_radix + j];
      }
      buf_write_pointers[current_radix] = 0;   // Reset pointer.
    }
  }

  // At the end of the scanning,
  // there may be some elements still in buffers which have not been flushed.
  // They need to be flushed.
  for (size_t current_radix=0; current_radix<num_radixes; ++current_radix) {
    // If the pointer is not at 0,
    // it means that this buffer contains some elements that have not been flushed.
    if (buf_write_pointers[current_radix]!=0) {
      size_t buf_pos_this_radix = current_radix * num_elems_per_radix_buf;
      for (size_t j=0; j<buf_write_pointers[current_radix]; ++j) {  // Flush buffer to memory.
        output_sorted[--histogram[current_radix]] = buffers[buf_pos_this_radix + j];
      }
    }
  }

  memcpy(array, output_sorted, array_size*sizeof(*array));
  free(output_sorted);
  free(array_radixes);
  free(buffers);
  free(buf_write_pointers);
  free(histogram);
}


/**
 * @param array : elements must greater or equal to zero
 * @param bits_leftmost_pos
 * @param bits_rightmost_pos
 *      Assume using "LSB 0" numbering,
 *      which means that bits numbering start at 0 for the least significant bit.
 *      So, bits_leftmost_pos > bits_rightmost_pos.
 */
void CountSortOnBits_Buffer_OMP(int *array, size_t array_size,
                                size_t bits_leftmost_pos, size_t bits_rightmost_pos,
                                size_t num_elems_per_radix_buf, size_t num_threads) {


  if ((array_size/num_threads)*num_threads != array_size) {
    fprintf(stderr,"For now, the algorithm only support the input that the array size is multiple of number of threads!\n");
    return;
  }

  omp_set_num_threads(num_threads);


  const size_t radix_size = bits_leftmost_pos-bits_rightmost_pos+1;
  const uint64_t num_radixes = (uint64_t ) 1 << radix_size;

  const size_t num_bits_to_shift = bits_rightmost_pos;
  const uint32_t mask = 0xFFFFFFFF >> (32-radix_size);

  const size_t num_elems_per_section = array_size/num_threads;

  int *output_sorted = malloc(array_size * sizeof(*output_sorted));

  // Get specific radixes of each element in array.
  int *array_radixes = malloc(array_size * sizeof(*array_radixes));
  #pragma omp parallel for
  for (size_t i=0; i<array_size; ++i) {
    array_radixes[i] = (array[i] >> num_bits_to_shift) & mask;
  }



  // Initialize local histograms to all zero.
  // Imagine this local histogram is a 2D array: [num_threads][num_radixes]
  // Each thread occupy one line.
  int *local_histograms = malloc(num_threads * num_radixes * sizeof(*local_histograms));
  memset(local_histograms, 0, num_threads*num_radixes*sizeof(*local_histograms));

  #pragma omp parallel
  {

    // Count and update the local histogram.
    // The shifting operation is because we are doing counting on specific bits.
    size_t current_thread_id = omp_get_thread_num();
    size_t array_section_start = current_thread_id * num_elems_per_section;
    size_t thread_histogram_start_pos = current_thread_id * num_radixes;
    for (size_t i = array_section_start; i < array_section_start + num_elems_per_section; ++i) {
      ++local_histograms[thread_histogram_start_pos + array_radixes[i]];
    }
  }

  // Calculate the prefix sum. The result is the ending offset of each radix.
  // According to the paper
  //   [Satish et al. - 2010 - Fast sort on CPUs and GPUs...]
  // But make some changes.
  // After calculating the prefix sum, the values of the histogram
  //   * indicate the STARTING write offset in the paper.
  //   * indicate the ENDING write offset in this code.
  int *updated_local_histograms = malloc(num_threads * num_radixes * sizeof(*updated_local_histograms));
  memset(updated_local_histograms, 0, num_threads*num_radixes*sizeof(*updated_local_histograms));
  #pragma omp parallel for
  for (size_t i=0; i<num_threads*num_radixes; ++i) {  // for each entry of histogram
    // calculate the prefix sum according to the formula in the paper and make some modification

    // Sum the number of elements whose radix are smaller than this.
    for (size_t j=0; j<num_threads; ++j) {
      for (size_t k=0; k<i%num_radixes; ++k) {
        updated_local_histograms[i] += local_histograms[j*num_radixes + k];
      }
    }

    // Sum the number of elements whose radix are the same as this
    // but were calculated by the threads whose ID are smaller than or equal to this.
    for (size_t j=0; j<=(size_t)(i/num_radixes); ++j) {
      updated_local_histograms[i] += local_histograms[j*num_radixes + i%num_radixes];
    }
  }

  memcpy(local_histograms, updated_local_histograms, num_threads * num_radixes * sizeof(*local_histograms));
  free(updated_local_histograms);

  // Fill elements into the output array, according the histogram.
  // For the stable sorting algorithm, counting sort now need to
  // scan the original array from the end to the start.
  // If scanning from the start, the output is unstable.
  #pragma omp parallel
  {
    // Initialize buffers. And set write pointers to all zero.
    int *buffers = malloc(num_radixes * num_elems_per_radix_buf * sizeof(*buffers));
    size_t *buf_write_pointers = malloc(num_radixes * sizeof(*buf_write_pointers));
    memset(buf_write_pointers, 0, num_radixes*sizeof(*buf_write_pointers));


    size_t current_thread_id = omp_get_thread_num();
    size_t array_section_start = current_thread_id * num_elems_per_section;
    size_t thread_histogram_start_pos = current_thread_id * num_radixes;
    // Note the STABILITY of the sorting algorithm!!!
    for (int i = array_section_start+num_elems_per_section-1; i >= (int)array_section_start; --i) {
      int current_radix = array_radixes[i];

      // Put element into buffer.
      size_t buf_w_p = buf_write_pointers[current_radix]++;
      buffers[current_radix * num_elems_per_radix_buf + buf_w_p] = array[i];

      // If the buffer is full, flush it.
      if (buf_write_pointers[current_radix]==num_elems_per_radix_buf) {
        size_t buf_pos_this_radix = current_radix * num_elems_per_radix_buf;
        for (size_t j=0; j<num_elems_per_radix_buf; ++j) {  // Flush buffer to memory.
          int write_pointer = --local_histograms[thread_histogram_start_pos + current_radix];
          output_sorted[write_pointer] = buffers[buf_pos_this_radix + j];
        }
        buf_write_pointers[current_radix] = 0;   // Reset pointer.
      }

    }

    // At the end of the scanning,
    // there may be some elements still in buffers which have not been flushed.
    // They need to be flushed.
    for (size_t current_radix=0; current_radix<num_radixes; ++current_radix) {
      // If the pointer is not at 0,
      // it means that this buffer contains some elements that have not been flushed.
      if (buf_write_pointers[current_radix]!=0) {
        size_t buf_pos_this_radix = current_radix * num_elems_per_radix_buf;
        for (size_t j=0; j<buf_write_pointers[current_radix]; ++j) {  // Flush buffer to memory.
          int write_pointer = --local_histograms[thread_histogram_start_pos + current_radix];
          output_sorted[write_pointer] = buffers[buf_pos_this_radix + j];
        }
      }
    }
    free(buffers);
    free(buf_write_pointers);
  }

  memcpy(array, output_sorted, array_size*sizeof(*array));
  free(output_sorted);
  free(local_histograms);
  free(array_radixes);
}



void RadixSortLSD_OMP(int *arr, size_t arr_size, size_t radix_size, size_t num_threads) {
  int num_digits = ((sizeof(int)*8) + radix_size - 1)/radix_size;  // Ceiling.
  if (num_threads==1) {
    for (size_t i=0; i<num_digits; ++i) {
      int left = i*radix_size+radix_size-1;
      left = left>=32 ? 31 : left;
      int right = i*radix_size;
      CountSortOnBits(arr, arr_size, left, right);
    }
  } else if (num_threads>1) {
    for (size_t i=0; i<num_digits; ++i) {
      int left = i*radix_size+radix_size-1;
      left = left>=32 ? 31 : left;
      int right = i*radix_size;
      CountSortOnBits_OMP(arr, arr_size,
                        left, right,
                        num_threads);
    }
  } else {
    exit(1);
  }

}


void RadixSortLSD_Buffer_OMP(int *arr, size_t arr_size, size_t radix_size, 
                             size_t num_elems_per_radix_buf,
                             size_t num_threads) {
  int num_digits = ((sizeof(int)*8) + radix_size - 1)/radix_size;  // Ceiling.

  if (num_threads==1) {
    for (size_t i=0; i<num_digits; ++i) {
      int left = i*radix_size+radix_size-1;
      left = left>=32 ? 31 : left;
      int right = i*radix_size;
      CountSortOnBits_Buffer_1Thread(arr, arr_size,
                                     left, right,
                                     num_elems_per_radix_buf);
    }
  } else if (num_threads>1) {
    for (size_t i=0; i<num_digits; ++i) {
      int left = i*radix_size+radix_size-1;
      left = left>=32 ? 31 : left;
      int right = i*radix_size;
      CountSortOnBits_Buffer_OMP(arr, arr_size,
                                 left, right,
                                 num_elems_per_radix_buf, num_threads);
    }
  } else {
    exit(1);
  }
}


void ParalMergeSort_SIMD() {}


#endif //SORT_ALGO_ACCELERATION_SORTALGOIMPLEACCEL_H
