#include<stdio.h>

int str_len(const char* s)  {
    int len = 0;
    while(*s != '\0')    {
        len++;
        s++;
    }
    return len;
}

char* str_cpy(char* dest, const char* src)  {
    char *ptr = dest;
    while ((*ptr++ = *src++) != '\0');
    return dest;
}

void str_cmp(const char* s1, const char* s2)    {
    while (*s1 && (*s1++ == *s2++));
    //see -1 here came because of post increment operator.
    return *(unsigned char*)(s1 - 1) - *(unsigned char*)(s2 - 1);
}

int main()  {
    char inp[7] = "koushu";
    char out[7];
    printf("%s\n", inp);
    printf("length of %s is %d\n",inp, str_len(inp));
    printf("Input String: %s\n",inp);
    str_cpy(out, inp);
    printf("Copied String: %s\n",out);
    return 0;
}