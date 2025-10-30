#include <stdio.h>
#include <stdlib.h>
#include "../../inc/uthash.h"

typedef struct hash_type {
    int id;             /* key */
    int val;            /* value */
    UT_hash_handle hh;  /* makes this structure hashable */
} hash_type;

/* Add an item */
void add_item(hash_type **items, int id, int val) {
    hash_type *s = malloc(sizeof(hash_type));
    s->id = id;
    s->val = val;
    HASH_ADD_INT(*items, id, s);
}

/* Find an item */
hash_type *find_item(hash_type *items, int id) {
    hash_type *s;
    HASH_FIND_INT(items, &id, s);
    return s;
}

/* Delete an item */
void delete_item(hash_type **items, hash_type *s) {
    HASH_DEL(*items, s);
    free(s);
}

/* Print all */
void print_items(hash_type *items) {
    hash_type *s;
    for (s = items; s != NULL; s = s->hh.next) {
        printf("id=%d, val=%d\n", s->id, s->val);
    }
}

/* Delete all */
void delete_all(hash_type **items) {
    hash_type *current, *tmp;
    HASH_ITER(hh, *items, current, tmp) {
        HASH_DEL(*items, current);
        free(current);
    }
}

int main() {
    hash_type *items1 = NULL;
    hash_type *items2 = NULL;

    /* Fill first hash */
    for (int i = 1; i <= 5; i++) {
        add_item(&items1, i, i * 10);
    }

    /* Fill second hash */
    for (int i = 6; i <= 10; i++) {
        add_item(&items2, i, i * 100);
    }

    printf("Items in table 1:\n");
    print_items(items1);

    printf("\nItems in table 2:\n");
    print_items(items2);

    /* Lookup in items1 */
    int lookup_id = 3;
    hash_type *found = find_item(items1, lookup_id);
    if (found)
        printf("\nFound id=%d → val=%d in items1\n", found->id, found->val);

    /* Cleanup */
    delete_all(&items1);
    delete_all(&items2);

    return 0;
}
