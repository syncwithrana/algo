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

hash_type* hash_get_item(hash_type *items, int id) {
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