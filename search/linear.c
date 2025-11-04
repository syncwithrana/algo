#include <stdio.h>

int linearSearch(int arr[], int n, int key) {
    for (int i = 0; i < n; i++) {
        if (arr[i] == key)
            return i;
    }
    return -1;
}

int main() {
    int arr[] = {5, 3, 8, 4, 2};
    int n = sizeof(arr) / sizeof(arr[0]);
    int key = 4;

    int idx = linearSearch(arr, n, key);

    if (idx != -1)
        printf("Element %d found at index %d\n", key, idx);
    else
        printf("Element %d not found\n", key);

    return 0;
}
