#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define T double
#define F 13
#define N_IN 5000000
#define N_REPEAT 5

void forest_root(T *, T *, int, int);

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  time_t t, t2;
  srand((unsigned) time(&t));
  T *in = malloc(N_IN * F * sizeof(T));
  T *out = malloc(N_IN * sizeof(T));
  for (size_t i = 0; i < N_IN * F; ++i) {
    in[i] = (T)rand()/((T)RAND_MAX/(T)50);
  }
  time(&t);
  for (size_t i = 0; i < N_REPEAT; ++i) {
    forest_root(in, out, 0, N_IN);
  }
  time(&t2);
  printf("%ld\n", t2 - t);
  printf("%lf\n", out[0]);
}
