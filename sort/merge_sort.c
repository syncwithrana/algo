#include <stdio.h>
#include <stdlib.h>
#include "utils.c"

void merge(int arr[], int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    int* L = copyArray(&arr[left], n1);
    int* R = copyArray(&arr[mid + 1], n2);

    int i = 0, j = 0;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[left++] = L[i++];
        }
        else {
            arr[left++] = R[j++];
        }
    }

    while (i < n1) {
        arr[left++] = L[i++];
    }

    while (j < n2) {
        arr[left++] = R[j++];
    }
}

void mergeSort(int arr[], int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

int main() {
    int arr[] = { 12, 11, 13, 5, 6, 7 };
    int n = sizeof(arr) / sizeof(arr[0]);
	
    mergeSort(arr, 0, n - 1);

    for (int i = 0; i < n; i++)
        printf("%d ", arr[i]);
    return 0;
}