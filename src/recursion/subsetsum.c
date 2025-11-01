#include <stdio.h>
int is_subset_sum(int arr[], int n, int sum) {
    if (sum == 0) return 1;
    if (sum < 0 || n == 0) return 0;
    int left = is_subset_sum(arr, n - 1, sum);
    int right = is_subset_sum(arr, n - 1, sum - arr[n - 1]);
    return left + right;
}

int main() {
    int arr[] = {3, 34, 4, 12, 5, 2};
    int sum = 9;
    int n = sizeof(arr) / sizeof(arr[0]);

    printf("is subset sum: %d\n", is_subset_sum(arr, n, sum));
    return 0;
}