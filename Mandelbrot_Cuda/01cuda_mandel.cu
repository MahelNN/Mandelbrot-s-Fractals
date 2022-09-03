#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "gpuerrchk.h"

/* Bounds of the Mandelbrot set */
#define X_MIN -1.78
#define X_MAX 0.78
#define Y_MIN -0.961
#define Y_MAX 0.961
/* numbers of threads per block */
#define NTHREADS 32

typedef struct {

  int nb_rows, nb_columns; /* Dimensions */
  char * pixels; /* Linearized matrix of pixels */

} Image;

static void error_options () {

  fprintf (stderr, "Use : ./mandel [options]\n\n");
  fprintf (stderr, "Options \t Meaning \t\t Default val.\n\n");
  fprintf (stderr, "-n \t\t Nb iter. \t\t 100\n");
  fprintf (stderr, "-b \t\t Bounds \t\t -1.78 0.78 -0.961 0.961\n");
  fprintf (stderr, "-d \t\t Dimensions \t\t 1024 768\n");
  fprintf (stderr, "-f \t\t File \t\t Image/mandel_cuda.ppm\n");
  exit (1);
}

static void analyzis (int argc, char * * argv, int * nb_iter, double * x_min, double * x_max, double * y_min, double * y_max, int * width, int * height, char * * path) {

  const char * opt = "b:d:n:f:" ;
  int c ;

  /* Default values */
  * nb_iter = 100;
  * x_min = X_MIN;
  * x_max = X_MAX;
  * y_min = Y_MIN;
  * y_max = Y_MAX;
  * width = 1024;
  * height = 768;
  
  * path = "Image/mandel_cuda.ppm";

  /* Analysis of arguments */
  while ((c = getopt (argc, argv, opt)) != EOF) {
    
    switch (c) {
      
    case 'b':
      sscanf (optarg, "%lf", x_min);
      sscanf (argv [optind ++], "%lf", x_max);
      sscanf (argv [optind ++], "%lf", y_min);
      sscanf (argv [optind ++], "%lf", y_max);
      break ;
    case 'd': /* width */
      sscanf (optarg, "%d", width);
      sscanf (argv [optind ++], "%d", height);
      break;
    case 'n': /* Number of iterations */
      * nb_iter = atoi (optarg);
      break;
    case 'f': /* Output file */
      * path = optarg;
      break;
    default :
      error_options ();
    };
  }  
}

static void initialization (Image * im, int nb_columns, int nb_rows) {
  im -> nb_rows = nb_rows;
  im -> nb_columns = nb_columns;
  im -> pixels = (char *) malloc (sizeof (char) * nb_rows * nb_columns); /* Space memory allocation */
} 

static void save (const Image * im, const char * path) {
  /* Image saving using the ASCII format'.PPM' */
  unsigned i;
  FILE * f = fopen (path, "w");  
  fprintf (f, "P6\n%d %d\n255\n", im -> nb_columns, im -> nb_rows); 
  for (i = 0; i < im -> nb_columns * im -> nb_rows; i ++) {
    char c = im -> pixels [i];
    fprintf (f, "%c%c%c", c, 2*c, c); /* Monochrome weight */
  }
  fclose (f);
}


__global__ static void    
cuda_Compute (char * pixels, int nb_rows, int nb_columns, int nb_iter, double x_min, double x_max, double y_min, double y_max) {
  
  unsigned int l = blockIdx.y * blockDim.y + threadIdx.y; /* Global indice line */
  unsigned int c = blockIdx.x * blockDim.x + threadIdx.x; /* Global indice column */
  unsigned int pos;
  double dx, dy, a, b, x, y;
  int i = 0;

  dx = (x_max - x_min) / nb_columns;
  dy = (y_max - y_min) / nb_rows; /* Discretization */

  if(l<nb_rows || l>0)
    if(c<nb_columns || c>0)
      {
	
	pos = l * nb_columns + c; /* Position of the computed pixel */
	/* Computation at each point of the image */
	a = x_min + c * dx;
	b = y_max - l * dy;
	x = 0, y = 0;      

	i=0;
	while (i < nb_iter) {
	  double tmp = x;
	  x = x * x - y * y + a;
	  y = 2 * tmp * y + b;
	  if (x * x + y * y > 4) /* Divergence ! */
	    break; 
	  else
	    i++;
	}     
	pixels[ pos++ ] = (double) i / nb_iter * 255;
      } 
}


double Choose_Tserial(int N)
{
  double Tserial;
  switch (N)
    {
    case 100:
      Tserial = 0.08354276;
      break;      
    case 200:
      Tserial = 0.15676422;
      break;      
    case 250:
      Tserial = 0.19308734;
      break;
    case 500:
      Tserial = 0.37361187; 
      break;
    case 750:
      Tserial = 0.55358624;
      break;
    case 1000:
      Tserial = 0.73245299;
      break;
    default:
      Tserial = 0.08354276;
      break;
    }
  return Tserial;
}


int main (int argc, char * * argv)
{
    
  int nb_iter, width, height; /* Degree of precision, dimensions of the image */  
  double x_min, x_max, y_min, y_max; /* Bounds of representation */
  char * path; /* File destination */
  Image im;
  char * im_d_pixels;

  struct timespec t0, t1, t2, t3;
  double Tserial, Tparallel, Tcomm;
  FILE *fptr;

  analyzis(argc, argv, & nb_iter, & x_min, & x_max, & y_min, & y_max, & width, & height, & path);
  initialization (& im, width, height); /* Initialisation on the Host */

  /* Initialization on the Device */
  cudaMalloc( & im_d_pixels, height * width * sizeof(char) );

  /* Mapping on the Device */
  dim3 gridDim ( ( width + NTHREADS - 1 ) / NTHREADS , ( height + NTHREADS-1 ) / NTHREADS ) ;
  dim3 blockDim ( NTHREADS, NTHREADS );

  /* Computation on the Device */
  clock_gettime(CLOCK_MONOTONIC, &t0);
  cuda_Compute <<< gridDim , blockDim >>> ( im_d_pixels, height, width, nb_iter, x_min, x_max, y_min, y_max );
  cudaDeviceSynchronize();
  clock_gettime(CLOCK_MONOTONIC, &t1);
  
  /* Copy data from Device to Host */
  clock_gettime(CLOCK_MONOTONIC, &t2);
  gpuErrchk( cudaMemcpy( im.pixels, im_d_pixels, height * width * sizeof( char ), cudaMemcpyDeviceToHost ) );
  clock_gettime(CLOCK_MONOTONIC, &t3);

  /* save as a picture */
  save (& im, path);

  Tparallel = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9f;
  Tcomm = (t3.tv_sec-t2.tv_sec)+(t3.tv_nsec-t2.tv_nsec)/1e9f;
   
  printf(" Tparallel = %2.9lf\n Tcomm = %2.9lf\n height=%d, width=%d \n", Tparallel, Tcomm, height, width);
      
  Tserial  = Choose_Tserial(nb_iter);  /* grisou-8 */
   
  fptr = fopen("Data/01speedup.dat","a"); 
  fprintf(fptr, "%2.9lf %2.9lf %2.9lf %2.9lf\n", Tserial, Tparallel+Tcomm, Tparallel, Tcomm);
  fclose(fptr);

  /* free the resources on the Device and Host */
  cudaFree( im_d_pixels );
  free( im.pixels );
  return 0 ;
}
