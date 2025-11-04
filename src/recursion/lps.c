// C program to find the Longest Palindromic Subsequence in a given string
#include <stdio.h>
#include <string.h>

int max(int x, int y) {
    return (x > y) ? x : y;
}

int lps(const char *s, int low, int high) {
    if (low > high) return 0;

    if (low == high)
        return 1;

    if (s[low] == s[high])
        return lps(s, low + 1, high - 1) + 2;
  
  	int a = lps(s, low, high - 1);
  	int b = lps(s, low + 1, high);
    return max(a, b);
}

int main() {
    char s[] = "bbabcbcab";
    printf("%d", lps(s, 0, strlen(s) - 1));
    return 0;
}