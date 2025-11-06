#include <stdio.h>
#include <string.h>

int is_palindrome(char* s) {
    int len = strlen(s);

    for (int i = 0; i < len / 2; i++) {
        if (s[i] != s[len - i - 1])  {
            return 0;
        }
    }
    return 1;
}

int main() {
    char s[] = "abbas";
    printf("%d\n", is_palindrome(s));
    return 0;
}