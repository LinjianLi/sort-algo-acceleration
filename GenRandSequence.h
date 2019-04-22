//
// Created by LinjianLi on 2019/4/19.
//

#ifndef SORT_ALGO_ACCELERATION_GENRANDSEQUENCE_H
#define SORT_ALGO_ACCELERATION_GENRANDSEQUENCE_H


/**
 * Initialize integer array d[N] with seed
 */
void sort_gen(int *d, int N, int seed) {
  srand(seed);
  for(int i=0;i<N;i++){
    d[i]=rand();
  }
}

#endif //SORT_ALGO_ACCELERATION_GENRANDSEQUENCE_H
