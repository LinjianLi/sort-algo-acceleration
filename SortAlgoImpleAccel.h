//
// Created by LinjianLi on 2019/4/19.
//

#ifndef SORT_ALGO_ACCELERATION_SORTALGOIMPLEACCEL_H
#define SORT_ALGO_ACCELERATION_SORTALGOIMPLEACCEL_H

#define SHUFFLE_REVERSE 0x1B
#define SHUFFLE_REVERSE_LAST_TW0 0xE1
#define SHUFFLE_REVERSE_FIRST_TW0 0xB4

#include <memory.h>
#include <string.h>  // memset
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <immintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <tmmintrin.h>
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


  // Initialize buffers. And set write pointers to the end of each buffer.
  // Program writes from the end to the start.
  int *buffers = malloc(num_radixes * num_elems_per_radix_buf * sizeof(*buffers));
  size_t *buf_write_pointers = malloc(num_radixes * sizeof(*buf_write_pointers));
  for (size_t i=0; i<num_radixes; ++i) {
    buf_write_pointers[i] = num_elems_per_radix_buf;
  }

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
            (--buf_write_pointers[current_radix])] = array[array_size-i];

    // If the buffer is full, flush it.
    if (buf_write_pointers[current_radix]==0) {
      size_t buf_pos_this_radix = current_radix * num_elems_per_radix_buf;

      // Flush buffer to memory.
      histogram[current_radix] -= num_elems_per_radix_buf;
      memcpy(output_sorted+histogram[current_radix], buffers+buf_pos_this_radix, num_elems_per_radix_buf*sizeof(*output_sorted));

      buf_write_pointers[current_radix] = num_elems_per_radix_buf;   // Reset pointer.
    }
  }

  // At the end of the scanning,
  // there may be some elements still in buffers which have not been flushed.
  // They need to be flushed.
  for (size_t current_radix=0; current_radix<num_radixes; ++current_radix) {
    // If the pointer is not at the end,
    // it means that this buffer contains some elements that have not been flushed.
    if (buf_write_pointers[current_radix]!=num_elems_per_radix_buf) {
      size_t buf_pos_this_radix = current_radix * num_elems_per_radix_buf;

      // Flush buffer to memory.
      int num_remain = num_elems_per_radix_buf - buf_write_pointers[current_radix];
      histogram[current_radix] -= num_remain;
      memcpy(output_sorted+histogram[current_radix],
              buffers + buf_pos_this_radix + buf_write_pointers[current_radix],
             num_remain*sizeof(*output_sorted));
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
    // Initialize buffers. And set write pointers to the end of each buffer.
    // Program writes from the end to the start.
    int *buffers = malloc(num_radixes * num_elems_per_radix_buf * sizeof(*buffers));
    size_t *buf_write_pointers = malloc(num_radixes * sizeof(*buf_write_pointers));
    for (size_t i=0; i<num_radixes; ++i) {
      buf_write_pointers[i] = num_elems_per_radix_buf;
    }

    size_t current_thread_id = omp_get_thread_num();
    size_t array_section_start = current_thread_id * num_elems_per_section;
    size_t thread_histogram_start_pos = current_thread_id * num_radixes;
    // Note the STABILITY of the sorting algorithm!!!
    for (int i = array_section_start+num_elems_per_section-1; i >= (int)array_section_start; --i) {
      int current_radix = array_radixes[i];

      // Put element into buffer.
      size_t buf_w_p = --buf_write_pointers[current_radix];
      buffers[current_radix * num_elems_per_radix_buf + buf_w_p] = array[i];

      // If the buffer is full, flush it.
      if (buf_write_pointers[current_radix]==0) {
        size_t buf_pos_this_radix = current_radix * num_elems_per_radix_buf;

//        for (size_t j=0; j<num_elems_per_radix_buf; ++j) {  // Flush buffer to memory.
//          int write_pointer = --local_histograms[thread_histogram_start_pos + current_radix];
//          output_sorted[write_pointer] = buffers[buf_pos_this_radix + j];
//        }

        // Flush buffer to memory.
        local_histograms[thread_histogram_start_pos + current_radix] -= num_elems_per_radix_buf;
        int write_pointer = local_histograms[thread_histogram_start_pos + current_radix];
        memcpy(output_sorted+write_pointer, buffers+buf_pos_this_radix, num_elems_per_radix_buf*sizeof(*output_sorted));

        buf_write_pointers[current_radix] = num_elems_per_radix_buf;   // Reset pointer.
      }

    }

    // At the end of the scanning,
    // there may be some elements still in buffers which have not been flushed.
    // They need to be flushed.
    for (size_t current_radix=0; current_radix<num_radixes; ++current_radix) {
      // If the pointer is not at 0,
      // it means that this buffer contains some elements that have not been flushed.
      if (buf_write_pointers[current_radix]!=num_elems_per_radix_buf) {
        size_t buf_pos_this_radix = current_radix * num_elems_per_radix_buf;


//        for (size_t j=0; j<buf_write_pointers[current_radix]; ++j) {  // Flush buffer to memory.
//          int write_pointer = --local_histograms[thread_histogram_start_pos + current_radix];
//          output_sorted[write_pointer] = buffers[buf_pos_this_radix + j];
//        }

        // Flush buffer to memory.
        int num_remain = num_elems_per_radix_buf - buf_write_pointers[current_radix];
        local_histograms[thread_histogram_start_pos + current_radix] -= num_remain;
        int write_pointer = local_histograms[thread_histogram_start_pos + current_radix];
        memcpy(output_sorted+write_pointer,
               buffers + buf_pos_this_radix + buf_write_pointers[current_radix],
               num_remain*sizeof(*output_sorted));

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


void Simple_Merge_2_Consecutive(int *arr, int start, int start2, int end_exclude) {
  int num_total = end_exclude - start;
  int *merged = malloc(num_total * sizeof(*merged));
  int wp = 0, rp1 = start, rp2 = start2;
  while (rp1<start2 && rp2<end_exclude) {
    merged[wp++] = arr[rp1]<arr[rp2] ? arr[rp1++] : arr[rp2++];
  }
  while (rp1<start2) {
    merged[wp++] = arr[rp1++];
  }
  while (rp2<end_exclude) {
    merged[wp++] = arr[rp2++];
  }
  memcpy(arr+start, merged, num_total*sizeof(*arr));
  free(merged);
}


void Merge_Radix_OMP(int *arr, size_t arr_size, size_t radix_size, size_t num_threads) {
  // todo: debug

  fprintf(stderr, "This function has not been debugged to "
                  "address the problem about boundary!!!");

  omp_set_num_threads(num_threads);
  int elements_per_thread = arr_size/num_threads;

  // Sort each block.
  #pragma omp parallel
  {
    RadixSortLSD_OMP(arr + elements_per_thread*omp_get_thread_num(),
                     elements_per_thread,
                     radix_size,
                     1);
  }

  // Merge all blocks. Assuming the number of blocks is power of 2.
  // todo: Address the problem when then number of blocks is not power of 2.
  int run_length = elements_per_thread;
  int num_remain_blocks = num_threads;
  while (num_remain_blocks>1) {
    #pragma omp parallel for
    for (int i=0; i<num_remain_blocks; i+=2) {
      int start_pos = i*run_length;
      Simple_Merge_2_Consecutive(arr, start_pos, start_pos+run_length, start_pos+2*run_length);
    }
    run_length <<= 1;
    num_remain_blocks >>= 1;
  }
}



void Sort4x4OnColumn(__m128i a, __m128i b, __m128i c, __m128i d) {
  __m128i temp_min0, temp_min1, temp_max0, temp_max1, temp_mid0, temp_mid1;
  temp_min0 = _mm_min_epi32(a, b);
  temp_min1 = _mm_min_epi32(c, d);
  temp_max0 = _mm_max_epi32(a, b);
  temp_max1 = _mm_max_epi32(c, d);
  a = _mm_min_epi32(temp_min0, temp_min1);
  d = _mm_min_epi32(temp_max0, temp_max1);
  temp_mid0 = _mm_max_epi32(temp_min0, temp_min1);
  temp_mid1 = _mm_min_epi32(temp_max0, temp_max1);
  b = _mm_min_epi32(temp_mid0, temp_mid1);
  c = _mm_max_epi32(temp_mid0, temp_mid1);
}


void Transpose4x4(__m128i a, __m128i b, __m128i c, __m128i d) {
  // Similar to the _MM_TRANSPOSE4_PS in <xmmintrin.h>.
  __m128i temp0, temp1, temp2, temp3;
  temp0 = _mm_unpacklo_epi32(a, b);
  temp2 = _mm_unpacklo_epi32(c, d);
  temp1 = _mm_unpackhi_epi32(a, b);
  temp3 = _mm_unpackhi_epi32(c, d);
  a = _mm_unpacklo_epi32(temp0, temp2);
  b = _mm_unpackhi_epi32(temp0, temp2);
  c = _mm_unpacklo_epi32(temp1, temp3);
  d = _mm_unpackhi_epi32(temp1, temp3);
}


// The server of the course do not support AVX2!
// Which means that "_mm_blend_epi32" can not be used.
// So, I did not finish the remaining lab.



//void InsertionSort(int *arr, size_t arr_size, int start, int end_exclude) {
//  for (int i=start+1; i<end_exclude; ++i) {
//    int curr = arr[i];
//    int j=i-1;
//    for (; j>=0 && arr[j]>curr; --j) {
//      arr[j+1] = arr[j];
//    }
//    arr[j+1] = curr;
//  }
//}
//
//
//void BitonicMergeKernel(__m128i *O1, __m128i *O2, __m128i A, __m128i B) {
//
//  __m128i L1, H1, L1p, H1p;
//  L1 = _mm_min_epi32(A, B);
//  H1 = _mm_max_epi32(A, B);
//  // In L1p, the first two ints are from L1, and the last two ints are from H1.
//  L1p = _mm_blend_epi32(L1, _mm_shuffle_epi32(H1,SHUFFLE_REVERSE), 0xC);
//  L1p = _mm_shuffle_epi32(L1p, SHUFFLE_REVERSE_LAST_TW0);
//  // In H1p, the first two ints are from H1, and the last two ints are from L1.
//  H1p = _mm_blend_epi32(_mm_shuffle_epi32(L1,SHUFFLE_REVERSE), H1, 0x3);
//  H1p = _mm_shuffle_epi32(H1p, SHUFFLE_REVERSE_FIRST_TW0);
//
//  __m128i L2, H2, L2p, H2p, L2pp, H2pp;
//  L2 = _mm_min_epi32(L2, H2);
//  H2 = _mm_min_epi32(L2, H2);
//  L2p = _mm_unpacklo_epi32(L2, H2);
//  H2p = _mm_unpackhi_epi32(L2, H2);
//  L2pp = _mm_blend_epi32(L2p, _mm_shuffle_epi32(H2p, SHUFFLE_REVERSE), 0xC);
//  L2pp = _mm_shuffle_epi32(L2pp, SHUFFLE_REVERSE_LAST_TW0);
//  H2pp = _mm_blend_epi32(_mm_shuffle_epi32(L2p, SHUFFLE_REVERSE), H2p, 0x3);
//  H2pp = _mm_shuffle_epi32(H2pp, SHUFFLE_REVERSE_FIRST_TW0);
//
//  __m128i L3, H3;
//  L3 = _mm_min_epi32(L2pp, H2pp);
//  H3 = _mm_max_epi32(L2pp, H2pp);
//  *O1 = _mm_blend_epi32(L3, _mm_shuffle_epi32(H3,SHUFFLE_REVERSE), 0xC);
//  *O1 = _mm_shuffle_epi32(*O1,SHUFFLE_REVERSE_LAST_TW0);
//  *O2 = _mm_blend_epi32(_mm_shuffle_epi32(L3,SHUFFLE_REVERSE), H3, 0x3);
//  *O2 = _mm_shuffle_epi32(*O2,SHUFFLE_REVERSE_FIRST_TW0);
//
//}
//
//
//void MergeConsecutive2Seqs(int *arr, int first_start, int second_start, int second_end_exclude) {
//  // todo: implement
//}
//
//
//void SortBlock(int *arr, size_t arr_size, int start, int end_exclude, int num_threads) {
//  omp_set_num_threads(num_threads);
//  int elements_each_thread = ((end_exclude-start)+num_threads-1)/num_threads;  // Ceiling.
//  #pragma omp parallel
//  {
//    int pid = omp_get_thread_num();
//    int offset = pid*elements_each_thread;
//    int thread_element_start = start + offset;
//    int thread_element_end = thread_element_start + elements_each_thread;
//    thread_element_end = thread_element_end>end_exclude ? end_exclude : thread_element_end;
//
//    // Generate sorted sequences of length 4.
//    for (int i=thread_element_start; i<thread_element_end; i+=16) {
//      int remain_elements = thread_element_end - i;
//      if (remain_elements>=16) {
//        __m128i a, b, c, d;
//        a = _mm_loadu_si128((__m128i*)arr+i);
//        b = _mm_loadu_si128((__m128i*)arr+i+4);
//        c = _mm_loadu_si128((__m128i*)arr+i+8);
//        d = _mm_loadu_si128((__m128i*)arr+i+16);
//        Sort4x4OnColumn(a, b, c, d);
//        Transpose4x4(a, b, c, d);
//        _mm_storeu_si128((__m128i*)arr+i, a);
//        _mm_storeu_si128((__m128i*)arr+i+4, b);
//        _mm_storeu_si128((__m128i*)arr+i+8, c);
//        _mm_storeu_si128((__m128i*)arr+i+12, d);
//      } else {
//        InsertionSort(arr, arr_size, i, i+remain_elements);
//      }
//    }
//
//    // Merge sequences of length 4 to 8, and 8 to 16, and so on.
//    int sequence_size;
//    for (sequence_size=4; sequence_size<=elements_each_thread; sequence_size<<=1) {
//      int num_sequences = (elements_each_thread+sequence_size-1)/sequence_size;
//
//      // Merge 2 sequences at a time.
//      for (int i=0; i<num_sequences; i+=2) {
//
////        MergeConsecutive2Seqs(arr,
////                arr+thread_element_start+i*sequence_size,
////                arr+thread_element_start+(i+1)*sequence_size,
////                arr+thread_element_start+(i+1)*sequence_size+sequence_size);
//
//        int num_seq_length_4 = sequence_size/4;
//
//
//        __m128i O1, O2;
//        __m128i A, B;
//        A = _mm_loadu_si128((__m128i*)arr+thread_element_start+i*sequence_size);
//        B = _mm_loadu_si128((__m128i*)arr+thread_element_start+(i+1)*sequence_size);
//        B = _mm_shuffle_epi32(B,SHUFFLE_REVERSE);
//        BitonicMergeKernel(&O1, &O2, A, B);
//        // todo: store O1 in the output array
//        O1 = O2;
//        // todo: scan A and B, read the smaller one in to O2
//
//
//        // If the number of sequences is not even,
//        // there remains one sequence un-merged.
//        if (num_sequences%2!=0) {
//
//        }
//
//
//      }
//
//    }
//    // If true, then there remains two sequences un-merged.
//    // The total size of two sequences is the number of elements allocated to this thread.
//    if (sequence_size!=(elements_each_thread<<1)) {
//      // todo: implement
//    }
//
//  }
//}
//
//
//void ParalMergeSort_SIMD(int *arr, size_t arr_size, int num_threads) {
//  // According to the paper of Chhugani et al..
//  // [Efficient implementation of sorting on multi-core SIMD CPU architecture]
//  int M = 32768; // M=cacahe_size/(2*element_size)=256KB/8B=32K
//  int num_block = (arr_size+M-1)/M;  // Ceiling.
//  int M_thread = (M+num_threads-1)/num_threads;  // Ceiling.
//
//
//  int *partially_sorted_blocks = malloc(arr_size * sizeof(partially_sorted_blocks));
//
//  for (int i=0; i<num_block; ++i) {
//    int start = i*M , end_exclude = start+M;
//    end_exclude = end_exclude>arr_size ? arr_size : end_exclude;
//    SortBlock(arr, arr_size, start, end_exclude, num_threads);
//  }
//
//  // todo: implement
//
//}
//
//
//void SSETest() {
//  int aaa[10] = {1,2,3,4,5,6,7,8,9,10};
//  PrintArray_Int(aaa, 10, 10);
//  __m128i A = _mm_loadu_si128((__m128i*)aaa),
//          B = _mm_loadu_si128((__m128i*)aaa+4),
//          temp;
//
//  temp = _mm_shuffle_epi32(A, SHUFFLE_REVERSE);
//  _mm_storeu_si128((__m128i*)aaa, temp);
//  PrintArray_Int(aaa, 10, 10);
//
//  _mm_storeu_si128((__m128i*)aaa, B);
//  PrintArray_Int(aaa, 10, 10);
//
//  _mm_storeu_si128((__m128i*)aaa, A);
//  temp = _mm_blend_epi32(A, B, 0x3);
//  _mm_storeu_si128((__m128i*)aaa, temp);
//  PrintArray_Int(aaa, 10, 10);
//
//  return;
//}


#endif //SORT_ALGO_ACCELERATION_SORTALGOIMPLEACCEL_H
