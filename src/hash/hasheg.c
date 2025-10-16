#include <stdio.h>
#include <stdlib.h>
#include "../../inc/uthash.h"

struct my_struct {
    int id;             /* key */
    int val;            /* value */
    UT_hash_handle hh;  /* makes this structure hashable */
};

/* Add an item */
void add_item(struct my_struct **items, int id, int val) {
    struct my_struct *s = malloc(sizeof(struct my_struct));
    s->id = id;
    s->val = val;
    HASH_ADD_INT(*items, id, s);
}

/* Find an item */
struct my_struct *find_item(struct my_struct *items, int id) {
    struct my_struct *s;
    HASH_FIND_INT(items, &id, s);
    return s;
}

/* Delete an item */
void delete_item(struct my_struct **items, struct my_struct *s) {
    HASH_DEL(*items, s);
    free(s);
}

/* Print all */
void print_items(struct my_struct *items) {
    struct my_struct *s;
    for (s = items; s != NULL; s = s->hh.next) {
        printf("id=%d, val=%d\n", s->id, s->val);
    }
}

/* Delete all */
void delete_all(struct my_struct **items) {
    struct my_struct *current, *tmp;
    HASH_ITER(hh, *items, current, tmp) {
        HASH_DEL(*items, current);
        free(current);
    }
}

int main() {
    struct my_struct *items1 = NULL;
    struct my_struct *items2 = NULL;

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
    struct my_struct *found = find_item(items1, lookup_id);
    if (found)
        printf("\nFound id=%d → val=%d in items1\n", found->id, found->val);

    /* Cleanup */
    delete_all(&items1);
    delete_all(&items2);

    return 0;
}
