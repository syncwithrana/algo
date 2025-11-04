/* Weighted Job Scheduling */
#include<stdio.h>
#include<stdlib.h>
#include "array_util.c"

int compare(const void *a, const void *b) { 
    return ((int*)a)[0] - ((int*)b)[0];
}

// int maxProfitRecur(int** jobs, int ind, int last) {
//     if (ind == jobs.size())
//         return 0;

//     int ans = maxProfitRecur(jobs, ind+1, last);

//     if(jobs[ind][0] >= last)
//         ans = max(ans, jobs[ind][2] + 
//         maxProfitRecur(jobs, ind+1, jobs[ind][1]));

//     return ans;
// }

// int maxProfit(int** jobs, int rows, int cols) {
//     int n = jobs.size();

//     sort(jobs.begin(), jobs.end());

//     return maxProfitRecur(jobs, 0, -1);
// }

int main() {
    int jobs[][3] = {
        {1, 2, 50}, 
        {3, 5, 20}, 
        {6, 19, 100}, 
        {2, 100, 200}
    };

    int rows = sizeof(jobs) / sizeof(jobs[0]);
    int cols = sizeof(jobs[0]) / sizeof(jobs[0][0]);
    qsort(jobs, rows, sizeof(int *), compare);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            printf("%d ", jobs[i][j]);
        printf("\n");
    }
    print_matrix(rows, cols, jobs);
    //printf("Weight Job Scheduling = %d\n", maxProfit(jobs, rows, cols));
    return 0;
}

int main()  {
     int rows = 4, cols = 3;
    
    int **arr = malloc(rows * sizeof(int *));
    for(int i = 0; i < rows; i++) {
        arr[i] = malloc(cols * sizeof(int));
    }
    
    // Initialize with sample data
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
    return 0;
}