#include<stdio.h>
#include<stdlib.h>
#include "array_util.c"

int compare_rows(const void *a, const void *b) {
    const int *row1 = *(const int **)a;
    const int *row2 = *(const int **)b;
    return row1[0] - row2[0];
}

int main()  {
    int rows = 4, cols = 3;

    int **arr = malloc(rows * sizeof(int *));
    for(int i = 0; i < rows; i++) {
        arr[i] = malloc(cols * sizeof(int));
    }

    int data[4][3] = {
        {1, 2, 50}, 
        {3, 5, 20}, 
        {6, 19, 100}, 
        {2, 100, 200}
    };

    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            arr[i][j] = data[i][j];
        }
    }

    qsort(arr, rows, sizeof(int *), compare_rows);
    print_dyn_matrix(rows, cols, arr);

    return 0;
}