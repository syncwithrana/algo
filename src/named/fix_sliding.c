#include<stdio.h>
int max(int num1, int num2){
    return (num1 > num2) ? num1 : num2;
}

int maxSum(int* arr, int n, int k) {
    if (n < k) {
        printf("Invalid");
        return -1;
    }

    int windowSum = 0;
    for (int i = 0; i < k; i++) {
        windowSum += arr[i];
    }

    int maxSum = windowSum;

    for (int i = k; i < n; i++) {
        windowSum += arr[i] - arr[i - k];
        maxSum = max(maxSum, windowSum);
    }

    return maxSum;
}

int main()  {
    int arr[] = {5, 2, -1, 0, 3};
    int size = sizeof(arr) / sizeof(arr[0]);
    int k = 3;
    printf("Max Sum with %d window = %d\n", k, maxSum(arr, size, k));
}