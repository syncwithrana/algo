#include<stdio.h>
#include<stdlib.h>

int* prefSum(int* arr, int n) {
    int* prefixSum = (int*)malloc(sizeof(int)*n);
    prefixSum[0] = arr[0];
    for (int i = 1; i < n; i++) {
        prefixSum[i] = prefixSum[i - 1] + arr[i];
    }
    return prefixSum;
}

int main()  {
    int arr[] = {10, 20, 10, 5, 15};
    int size = sizeof(arr) / sizeof(arr[0]);
    int* prefixSum = prefSum(arr, size);
    for (int i = 0; i < size; i++) {
        printf("%d ", prefixSum[i]);
    }
}