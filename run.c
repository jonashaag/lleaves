#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define T double
#define F 5
#define n_in 50000000

void forest_root(T *, T *, int, int);

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  time_t t, t2;
  srand((unsigned) time(&t));
  T *in = malloc(n_in * F * sizeof(T));
  T *out = malloc(n_in * sizeof(T));
  for (size_t i = 0; i < n_in * F; ++i) {
    in[i] = (T)rand()/((T)RAND_MAX/(T)50);
  }
  time(&t);
  forest_root(in, out, 0, n_in);
  time(&t2);
  printf("%ld\n", t2 - t);
  printf("%lf\n", out[0]);
}
