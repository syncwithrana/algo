#include <stdio.h>
#include <stdlib.h>
#include "../../inc/uthash.h"

typedef struct int_set {
    int key;
    UT_hash_handle hh;
} int_set;

void add_key(int_set **set, int key) {
    int_set *s;
    HASH_FIND_INT(*set, &key, s);
    if (s) return; // already present
    s = malloc(sizeof(int_set));
    s->key = key;
    HASH_ADD_INT(*set, key, s);
}

int has_key(int_set *set, int key) {
    int_set *s;
    HASH_FIND_INT(set, &key, s);
    return s != NULL;
}

void delete_key(int_set **set, int key) {
    int_set *s;
    HASH_FIND_INT(*set, &key, s);
    if (s) {
        HASH_DEL(*set, s);
        free(s);
    }
}

void print_keys(int_set *set) {
    int_set *s, *tmp;
    HASH_ITER(hh, set, s, tmp) {
        printf("key=%d\n", s->key);
    }
}

void delete_all(int_set **set) {
    int_set *current, *tmp;
    HASH_ITER(hh, *set, current, tmp) {
        HASH_DEL(*set, current);
        free(current);
    }
}

int main(void) {
    int_set *set = NULL;

    for (int i = 1; i <= 5; i++) add_key(&set, i);

    printf("Keys in set:\n");
    print_keys(set);

    int k = 3;
    printf("\nChecking key %d: %s\n", k, has_key(set, k) ? "present" : "absent");

    printf("\nDeleting key %d\n", k);
    delete_key(&set, k);

    printf("\nKeys after deletion:\n");
    print_keys(set);

    delete_all(&set);
    return 0;
}