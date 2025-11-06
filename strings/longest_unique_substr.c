#include <stdio.h>
#include <string.h>
#include <stdbool.h>


bool getVisited(char c, bool vis[]) {
    return vis[c - 'a']; 
}

void setVisited(char c, bool vis[]) {
    vis[c - 'a'] = true;
}

void clearVisited(char c, bool vis[]) {
    vis[c - 'a'] = false; 
}

int longestUniqueSubstr(char* s) {
    int len = strlen(s);
    if (len == 0 || len == 1) return len;
    
    int res = 0;
    bool vis[26] = { false };

    int left = 0, right = 0;
    while (right < len) {

        while (getVisited(s[right], vis) == true) {
            clearVisited(s[left], vis);
            left++;
        }
        setVisited(s[right], vis);

        if ((right - left + 1) > res)   {
            res = right - left + 1;
        }
        right++;
    }
    return res;
}

int main() {
    char s[] = "geeksforgeeks";
    printf("length of longest Unique substr=%d\n", longestUniqueSubstr(s));
    return 0;
}