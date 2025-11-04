#include <stdio.h>

int is_subset_sum(int arr[], int n, int sum) {
    if (sum == 0) return 1;
    if (sum < 0 || n == 0) return 0;
    int left = is_subset_sum(arr, n - 1, sum);
    int right = is_subset_sum(arr, n - 1, sum - arr[n - 1]);
    return left || right;
}

int equal_partition(int arr[], int n) {
    int sum = 0;
    for(int i=0; i<n; i++)
        sum += arr[i];

    if (sum % 2 != 0)
        return 0;

    return is_subset_sum(arr, n, sum / 2);
}

int main() {
    int arr[] = { 1, 5, 11, 5};
    int size = sizeof(arr) / sizeof(arr[0]);
    printf("%d", equal_partition(arr, size));
    return 0;
}