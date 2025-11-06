#include <stdio.h>
#include <string.h>
#include "string_utils.c"

void reverseString(char* str) {
    int len = strlen(str);
    for (int i = 0; i < len / 2; i++) {
        swap(&str[i], &str[len - i - 1]);
    }
}

int main() {
    char s[] = "embedded";
    reverseString(s);
    printf("Reversed: %s\n", s);
    return 0;
}