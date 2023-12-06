#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "pgmio.h"

#define M 225
#define N 225

double gaussian_kernel[3][3] = {
    {1.0 / 16, 2.0 / 16, 1.0 / 16},
    {2.0 / 16, 4.0 / 16, 2.0 / 16},
    {1.0 / 16, 2.0 / 16, 1.0 / 16}
};

int main(int argc, char **argv)
{
    int i, j, k, l;
    float masterbuf[M][N];

    char *filename;
    char input[] = "image225x225.pgm";
    char output[] = "image-output225x225.pgm";
    filename = input;

    double start_time_total, end_time_total;
    double start_time_gaussian, end_time_gaussian;

    start_time_total = omp_get_wtime();
    pgmread(filename, masterbuf, M, N);

    printf("width: %d \nheight: %d\n", M, N);

    // Set the number of threads
    omp_set_num_threads(4);

    start_time_gaussian = omp_get_wtime();

    float image[M][N];

    #pragma omp parallel private(i, j, k, l) shared(masterbuf)
    {
        #pragma omp for schedule(static)
        for (i = 1; i < M - 1; i++) {
            for (j = 1; j < N - 1; j++) {
                float weighted_sum = 0.0;
                for (k = -1; k <= 1; k++) {
                    for (l = -1; l <= 1; l++) {
                        weighted_sum += masterbuf[i + k][j + l] * gaussian_kernel[k + 1][l + 1];
                    }
                }
                image[i][j] = weighted_sum;
            }
        }
    }

    end_time_gaussian = omp_get_wtime();

    printf("Finished\n");

    filename = output;
    printf("Output: <%s>\n", filename);
    pgmwrite(filename, image, M, N);

    end_time_total = omp_get_wtime();

    double total = (end_time_gaussian - start_time_gaussian);
    printf("Total Parallel Time: %fs\n", total);
    printf("Total Serial Time: %fs\n", (end_time_total - start_time_total) - total);
    printf("Total Time: %fs\n", end_time_total - start_time_total);

    return 0;
}