#include<stdio.h>
#include"string_utils.c"

int str_len(const char *s) {
    int i;
    for (i = 0; s[i] != '\0'; i++);
    return i;
}

char* str_cpy(char *dest, const char *src) {
    char *ptr = dest;
    while ((*ptr++ = *src++) != '\0');
    return dest;
}

int str_cmp(const char *s1, const char *s2) {
    while (*s1 && (*s1++ == *s2++));
    //see -1 here came because of post increment operator.
    return *(unsigned char *)(s1 - 1) - *(unsigned char *)(s2 - 1);
}

int str_str(const char *haystack, const char *needle) {
    int hlen = str_len(haystack);
    int nlen = str_len(needle);

    for (int i = 0; i <= hlen - nlen; i++) {
        int j;
        for (j = 0; j < nlen; j++) {
            if (haystack[i + j] != needle[j]) {
                break;
            }
        }
        if (j == nlen) {
            return i; // Found at index i
        }
    }
    return -1; // Not found
}

char* str_rev(char* str) {
    int len = str_len(str);
    for (int i = 0; i < len / 2; i++) {
        swap(&str[i], &str[len - i - 1]);
    }
    return str;
}


int main() {
    char s1[100] = "embedded";
    char s2[100] = "system";
    char s3[100];

    // strrev
    str_rev(s1);
    printf("Reversed s1: %s\n", s1);
    str_rev(s2);
    printf("Reversed s2: %s\n", s2);

    // strcpy
    str_cpy(s3, "hello");
    printf("Copied string 1: %s\n", s3);
    str_cpy(s3, s2);
    printf("Copied string 2: %s\n", s3);

    // strstr
    char text[] = "this is embedded system";
    printf("Found 'embedded' at: %d\n", str_str(text, "embedded"));
    printf("Found 'system' at: %d\n",str_str(text, "system"));

    // strlen
    printf("Length of 'hello': %d\n", str_len("hello"));
    printf("Length of s2: %d\n", str_len(s2));

    // strcmp
    printf("Compare 'abc' vs 'abc': %d\n", str_cmp("abc", "abc"));
    printf("Compare 'abc' vs 'abd': %d\n", str_cmp("abc", "abd"));

    return 0;
}
