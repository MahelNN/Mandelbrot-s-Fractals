#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <omp.h>

/* Bounds of the Mandelbrot set */
#define X_MIN -1.78
#define X_MAX 0.78
#define Y_MIN -0.961
#define Y_MAX 0.961

typedef struct
{

  int nb_rows, nb_columns; /* Dimensions */
  char *pixels;            /* Linearized matrix of pixels */

} Image;

static void error_options()
{

  fprintf(stderr, "Use : ./mandel [options]\n\n");
  fprintf(stderr, "Options \t Meaning \t\t Default val.\n\n");
  fprintf(stderr, "-n \t\t Nb iter. \t\t 100\n");
  fprintf(stderr, "-b \t\t Bounds \t\t -1.78 0.78 -0.961 0.961\n");
  fprintf(stderr, "-d \t\t Dimensions \t\t 1024 768\n");
  fprintf(stderr, "-f \t\t File \t\t Image/mandel_openmp.ppm\n");
  exit(1);
}

static void analyzis(int argc, char **argv, int *nb_iter, double *x_min, double *x_max, double *y_min, double *y_max, int *width, int *height, char **path)
{

  const char *opt = "b:d:n:f:";
  int c;

  /* Default values */
  *nb_iter = 100;
  *x_min = X_MIN;
  *x_max = X_MAX;
  *y_min = Y_MIN;
  *y_max = Y_MAX;
  *width = 1024;
  *height = 768;
  *path = "Image/mandel_openmp.ppm";

  /* Analysis of arguments */
  while ((c = getopt(argc, argv, opt)) != EOF)
  {

    switch (c)
    {

    case 'b':
      sscanf(optarg, "%lf", x_min);
      sscanf(argv[optind++], "%lf", x_max);
      sscanf(argv[optind++], "%lf", y_min);
      sscanf(argv[optind++], "%lf", y_max);
      break;
    case 'd': /* width */
      sscanf(optarg, "%d", width);
      sscanf(argv[optind++], "%d", height);
      break;
    case 'n': /* Number of iterations */
      *nb_iter = atoi(optarg);
      break;
    case 'f': /* Output file */
      *path = optarg;
      break;
    default:
      error_options();
    };
  }
}

static void initialization(Image *im, int nb_columns, int nb_rows)
{
  im->nb_rows = nb_rows;
  im->nb_columns = nb_columns;
  im->pixels = (char *)malloc(sizeof(char) * nb_rows * nb_columns); /* Space memory allocation */
}

static void save(const Image *im, const char *path)
{
  /* Image saving using the ASCII format'.PPM' */
  unsigned i;
  FILE *f = fopen(path, "w");
  fprintf(f, "P6\n%d %d\n255\n", im->nb_columns, im->nb_rows);
  for (i = 0; i < im->nb_columns * im->nb_rows; i++)
  {
    char c = im->pixels[i];
    fprintf(f, "%c%c%c", 3 * c, 1 * c, 2 * c); /* Monochrome weight */
  }
  fclose(f);
}

static void Compute(Image *im, int nb_iter, double x_min, double x_max, double y_min, double y_max)
{

  int pos = 0;

  int l, c, i = 0;
  double dx = (x_max - x_min) / im->nb_columns, dy = (y_max - y_min) / im->nb_rows; /* Discretization */

#pragma omp parallel for
  for (l = 0; l < im->nb_rows; l++)
  {

    for (c = 0; c < im->nb_columns; c++)
    {

      /* Computation at each point of the image */

      double a = x_min + c * dx, b = y_max - l * dy, x = 0, y = 0;
      i = 0;

      while (i < nb_iter)
      {
        double tmp = x;
        x = x * x - y * y + a;
        y = 2 * tmp * y + b;
        if (x * x + y * y > 4) /* Divergence ! */
          break;
        else
          i++;
      }
      pos = l * (im->nb_columns) + c;
      im->pixels[pos] = (double)i / nb_iter * 255;
    }
  }
}


double Choose_Tserial(int N)
{
  double Tserial;
  switch (N)
  {
  case 100:
    Tserial = 0.21072665;
    break;
  case 250:
    Tserial = 0.48865592;
    break;
  case 500:
    Tserial = 0.94984049;
    break;
  case 750:
    Tserial = 1.40573335;
    break;
  case 1000:
    Tserial = 1.86157966;
    break;

  default:
    Tserial = 0.21072665;
    break;
  }
  return Tserial;
}

int main(int argc, char **argv)
{

  int nb_iter, width, height;        /* Degree of precision, dimensions of the image */
  double x_min, x_max, y_min, y_max; /* Bounds of representation */
  char *path;                        /* File destination */
  Image im;

  FILE *fptr;
  struct timespec t0, t1;
  double Tserial, Tparallel, S, E;

  analyzis(argc, argv, &nb_iter, &x_min, &x_max, &y_min, &y_max, &width, &height, &path);
  initialization(&im, width, height); /* Image initiatlization */

  clock_gettime(CLOCK_MONOTONIC, &t0);               /* Timer */
  Compute(&im, nb_iter, x_min, x_max, y_min, y_max); /* Comtutation */
  clock_gettime(CLOCK_MONOTONIC, &t1);
  save(&im, path); /* Saving data */

  Tparallel = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9f;

  Tserial = Choose_Tserial(nb_iter);
  S = Tserial / Tparallel;
  E = S / omp_get_max_threads();

  fptr = fopen("Data/00speedup.dat", "a"); //"a" if the file doesnt exit creat it
  fprintf(fptr, "%d %2.9lf %2.9lf %2.9lf \n", omp_get_max_threads(), S, E, Tparallel);
  fclose(fptr);

  return 0;
}
