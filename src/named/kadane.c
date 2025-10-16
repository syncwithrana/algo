#include <stdio.h>
#include <limits.h>

long long maxSubarraySum(int arr[], int size) {
    long long res = arr[0];
    long long maxEnding = arr[0];

    for (int i = 1; i < size; i++) {
        maxEnding = (maxEnding + arr[i] > arr[i]) ? maxEnding + arr[i] : arr[i];
      
        if (res < maxEnding)    {
            res = maxEnding;
        }
    }
    return res;
}

int main() {
    int arr[] = {2, 3, -8, 7, -1, 2, 3};
    int size = sizeof(arr) / sizeof(arr[0]);
    printf("%lld\n", maxSubarraySum(arr, size));
    return 0;
}

/*
arr   maxEnding
2       2
3       5    
-8      -3
7       7
-1      6
2       8
3       11
*/