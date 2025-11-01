#include <stdio.h>

int max(int a, int b) {
    return (a > b) ? a : b;
}

int maxLoot(int *hval, int n) {
    if (n <= 0)  return 0;
    if (n == 1)  return hval[0];

    int pick = hval[n - 1] + maxLoot(hval, n - 2);
    int notPick = maxLoot(hval, n - 1);

    return max(pick, notPick);
}

int main() {
    int hval[] = {6, 7, 1, 3, 8, 2, 4};
    int n = sizeof(hval) / sizeof(hval[0]);
    printf("Max loot value = %d\n", maxLoot(hval, n));
    return 0;
}