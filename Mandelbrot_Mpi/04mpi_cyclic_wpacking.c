#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>

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
  fprintf(stderr, "-f \t\t File \t\t Image/04mandel_cyclic_wpacking.ppm\n");
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
  *path = "Image/04mandel_cyclic_wpacking.ppm";

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
    fprintf(f, "%c%c%c", c, 2 * c, 3 * c); /* Monochrome weight */
  }
  fclose(f);
}

static void Compute(Image *im, int nb_iter, double x_min, double x_max, double y_min, double y_max)
{

  int pos = 0;

  int l, c, i = 0;

  double dx = (x_max - x_min) / im->nb_columns, dy = (y_max - y_min) / im->nb_rows; /* Discretization */

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

      im->pixels[pos++] = (double)i / nb_iter * 255;
    }
  }
}

int main(int argc, char **argv)
{

  int nb_iter, width, height;        /* Degree of precision, dimensions of the image */
  double x_min, x_max, y_min, y_max; /* Bounds of representation */
  char *path;                        /* File destination */
  Image im, imloc;

  int size, rank, imlocSize, hloc, *tag, batch, l;
  double dy, y_min_loc, y_max_loc;     /* Local image arguments */
  double t0, Tparallel, Tserial, S, E; /* time and speedup variables */

  char buffer[1 << 20]; /* Large block of memory for packing */
  int position = 0;     /* Position inside the inside the buffer */

  FILE *fptr; /* data safe pointer */
  MPI_Status status;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size); /* number of processors */
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); /* processor unit */

  analyzis(argc, argv, &nb_iter, &x_min, &x_max, &y_min, &y_max, &width, &height, &path);
  initialization(&im, width, height); /* image initialization */

  hloc = 1;                        /* height of the local image */
  dy = (y_max - y_min) / height;   /* vertical step size */
  imlocSize = (int)(width * hloc); /* size of the local image */

  initialization(&imloc, width, hloc); /* local image initialization */
  tag = malloc(sizeof(int) * size);

  for (int i = 0; i < size; i++) /* Identify every package send at the reception */
    tag[i] = (i + 1) * 100;

  batch = height / size; /* block of size lines */

  /* Stop the program if processor number is not a divider of height */
  if (height / size * size != height)
  {
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Division %d/%d  not an integer!!\n", height, size);
    MPI_Finalize();
    exit(EXIT_SUCCESS);
  }

  /* Else ... */
  t0 = MPI_Wtime(); /* timer */

  for (int indexBatch = 0; indexBatch < batch; indexBatch++)
  {
    if (rank != 0)
    { /* Each compute and pack one a set of lines */

      y_max_loc = y_max - dy * (rank + size * indexBatch);
      y_min_loc = y_max_loc - dy;

      /* Computation of one line of pixels */
      Compute(&imloc, nb_iter, x_min, x_max, y_min_loc, y_max_loc);
      /* save the pixels vector inside the buffer to be sent */
      MPI_Pack(&(imloc.pixels[0]), imlocSize + 1, MPI_CHAR, buffer, 1 << 20, &position, MPI_COMM_WORLD);
    }
    else if (rank == 0)
    { /* The master process does't need to send its data */

      y_max_loc = y_max - dy * (size * indexBatch);
      y_min_loc = y_max_loc - dy;

      Compute(&imloc, nb_iter, x_min, x_max, y_min_loc, y_max_loc);
      l = (indexBatch * size) * imlocSize;
      for (int i = 0; i < width; i++)
        im.pixels[l + i] = imloc.pixels[i];
    }
  }

  if (rank != 0) /* Each cpu send the packed lines */
    MPI_Send(buffer, 1 << 20, MPI_PACKED, 0, tag[rank], MPI_COMM_WORLD);
  else if (rank == 0)
  { /* The master process receive the packed lines */

    for (int r = 1; r < size; r++)
    {

      MPI_Recv(buffer, 1 << 20, MPI_PACKED, r, tag[r], MPI_COMM_WORLD, &status);
      /* Reinitialize the position inside the buffer at each loop */
      position = 0;
      for (int indexBatch = 0; indexBatch < batch; indexBatch++)
      { /* Unpack the data */

        l = (indexBatch * size + r) * imlocSize;
        MPI_Unpack(buffer, 1 << 20, &position, &(im.pixels[l]), imlocSize + 1, MPI_CHAR, MPI_COMM_WORLD);
      }
    }
  }
  Tparallel = MPI_Wtime() - t0;

  if (rank == 0)
  {

    Tserial = 0.109751485;    /* Best Tserial of out of 10 runs (Grizou) */
    S = Tserial / Tparallel; /* Speedup */
    E = S / size;            /* Efficiency */
    /* saving data */
    fptr = fopen("Data/04speedup_mpi_cyclic_wpacking.dat", "a");
    fprintf(fptr, "%d %2.9lf %2.9lf %2.9lf \n", size, S, E, Tparallel);
    fclose(fptr);
    /* save the image */
    save(&im, path);
  }

  MPI_Finalize();
}
