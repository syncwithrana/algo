#include <stdio.h>
#include <stdlib.h>
#include "../../inc/uthash.h"

typedef struct hash_type {
    int id;
    int val;
    UT_hash_handle hh;
} hash_type;

void hash_add_item(hash_type **items, int id, int val) {
    hash_type *s = malloc(sizeof(hash_type));
    s->id = id;
    s->val = val;
    HASH_ADD_INT(*items, id, s);
}

hash_type* hash_find_item(hash_type *items, int id) {
    hash_type *s;
    HASH_FIND_INT(items, &id, s);
    return s;
}

void hash_delete_all(hash_type **items) {
    hash_type *current, *tmp;
    HASH_ITER(hh, *items, current, tmp) {
        HASH_DEL(*items, current);
        free(current);
    }
}

void hash_delete_item(hash_type **items, hash_type *s) {
    HASH_DEL(*items, s);
    free(s);
}

void hash_print_all(hash_type *items) {
    hash_type *s, *tmp;
    HASH_ITER(hh, items, s, tmp) {
        printf("id=%d, val=%d\n", s->id, s->val);
    }
}

int main() {
    hash_type *items1 = NULL;
    for (int i = 1; i <= 5; i++) {
        hash_add_item(&items1, i, i * 10);
    }

    printf("Items in table 1:\n");
    hash_print_all(items1);

    int lookup_id = 3;
    hash_type *found = hash_find_item(items1, lookup_id);
    if (found)
        printf("\nFound id=%d → val=%d in items1\n", found->id, found->val);

    hash_delete_all(&items1);

    return 0;
}
