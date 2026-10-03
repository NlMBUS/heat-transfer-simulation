
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <cuda_runtime.h>

#define epsilon 0.01
#define IDX(i, j, COLS) (((size_t)(i) * (size_t)(COLS)) + (size_t)(j))

__global__ void pixel_update_gpu(double *grid, double *updated, int y, int x) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; //column
    int i = blockIdx.y * blockDim.y + threadIdx.y; //row
    
    if (i >= y || j >= x) return;

    if ((j >= x/3 && j <= 1 + x*2/3 && i <= 1 + y/12) || 
        (j <= x/20 || j >= x*19/20 || i <= y/20 || i >= y*19/20)) {
        updated[IDX(i, j, x)] = grid[IDX(i, j, x)];
        return;
    }

    //interior cell
    double up = grid[IDX(i-1, j, x)];
    double down = grid[IDX(i+1, j, x)];
    double left = grid[IDX(i, j-1, x)];
    double right = grid[IDX(i, j+1, x)];
    updated[IDX(i, j, x)] = (up + down + left + right) / 4.0;
}

void pixel_init(int y, int x, double *grid) {
    for (int i = 0; i < y; i++) {
        for (int j = 0; j < x; j++) {
            if (j >= x/3 && j <= 1 + x*2/3 && i <= 1 + y/12) {
                grid[IDX(i, j, x)] = 100; //set heater temp and colour
            }
            else if (j <= x/20 || j >= x*19/20 || i <= y/20 || i >= y*19/20) {
                grid[IDX(i, j, x)] = 20; //set wall temp and colour
            }
            else {
                grid[IDX(i, j, x)] = -20; //set room temp
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 8) {
        printf("Usage: %s <rows> <columns> <iterations> <blockX> <blockY> <gridX> <gridY>\n", argv[0]);
        printf("  rows, columns: dimensions of the classroom grid\n");
        printf("  iterations: number of iterations to perform\n");
        printf("  blockX, blockY: block dimensions (e.g., 16 16)\n");
        printf("  gridX, gridY: grid dimensions (use 0 for automatic calculation)\n");
        exit(1);
    }

    int y = atoi(argv[1]);
    int x = atoi(argv[2]);
    int max_iteration = atoi(argv[3]);
    int block_x = atoi(argv[4]);
    int block_y = atoi(argv[5]);
    int grid_x = atoi(argv[6]);
    int grid_y = atoi(argv[7]);

    //auto-calculate grid dimensions if set to 0
    if (grid_x == 0) {
        grid_x = (x + block_x - 1) / block_x;
    }
    if (grid_y == 0) {
        grid_y = (y + block_y - 1) / block_y;
    }

    size_t N = (size_t)y * (size_t)x;
    double *grid = (double *)malloc(N * sizeof(double));

    pixel_init(y, x, grid);

    double *d_grid, *d_updated;
    cudaMalloc(&d_grid, N * sizeof(double));
    cudaMalloc(&d_updated, N * sizeof(double));
    cudaMemcpy(d_grid, grid, N * sizeof(double), cudaMemcpyHostToDevice);

    dim3 block(block_x, block_y);
    dim3 gridDim(grid_x, grid_y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int iteration = 0; iteration < max_iteration; iteration++) {
        pixel_update_gpu<<<gridDim, block>>>(d_grid, d_updated, y, x);
        cudaDeviceSynchronize();

        //swap pointers
        double *tmp = d_grid;
        d_grid = d_updated;
        d_updated = tmp;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    cudaMemcpy(grid, d_grid, N * sizeof(double), cudaMemcpyDeviceToHost);

    //write temperatures
    FILE *fp = fopen("mat-parallel.dat", "w");
    for (int i = 0; i < y; i++) {
        for (int j = 0; j < x; j++) {
            fprintf(fp, "%.2f\t", grid[IDX(i, j, x)]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);

    printf("Total GPU time: %.2f\n", elapsed_ms);

    cudaFree(d_grid);
    cudaFree(d_updated);
    free(grid);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
