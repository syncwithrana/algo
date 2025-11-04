#include <stdio.h>
#include <stdlib.h>

void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

void printArray(int arr[], int n) {
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

int* copyArray(int arr[], int size) {
    int* newArr = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        newArr[i] = arr[i];
    }
    return newArr;
}