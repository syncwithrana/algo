// restaurant_queue.c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

#define MAX_TRAYS 3
#define ORDERS 10

int trays[MAX_TRAYS];
int count = 0, in = 0, out = 0;

pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t not_full  = PTHREAD_COND_INITIALIZER;
pthread_cond_t not_empty = PTHREAD_COND_INITIALIZER;

void *chef(void *arg) {
    for (int i = 1; i <= ORDERS; ++i) {
        pthread_mutex_lock(&lock);

        while (count == MAX_TRAYS) {
            printf("👨‍🍳 Chef: trays full, waiting...\n");
            pthread_cond_wait(&not_full, &lock);
        }

        trays[in] = i;
        in = (in + 1) % MAX_TRAYS;
        count++;

        printf("👨‍🍳 Chef cooked order #%d (trays=%d)\n", i, count);
        pthread_cond_signal(&not_empty);
        pthread_mutex_unlock(&lock);

        sleep(1); // simulate cooking time
    }
    return NULL;
}

void *waiter(void *arg) {
    for (int i = 1; i <= ORDERS; ++i) {
        pthread_mutex_lock(&lock);

        while (count == 0) {
            printf("🧑‍🍽️ Waiter: no dishes, waiting...\n");
            pthread_cond_wait(&not_empty, &lock);
        }

        int order = trays[out];
        out = (out + 1) % MAX_TRAYS;
        count--;

        printf("🧑‍🍽️ Waiter served order #%d (trays=%d)\n", order, count);
        pthread_cond_signal(&not_full);
        pthread_mutex_unlock(&lock);

        sleep(2); // simulate serving time
    }
    return NULL;
}

int main() {
    pthread_t tchef, twaiter;

    pthread_create(&tchef, NULL, chef, NULL);
    pthread_create(&twaiter, NULL, waiter, NULL);

    pthread_join(tchef, NULL);
    pthread_join(twaiter, NULL);

    printf("\n✅ All orders served. Kitchen closed!\n");
    return 0;
}
