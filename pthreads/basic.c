#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>
#include <sys/syscall.h>

// ===========================================
// SECTION 1: Basic thread creation and joining
// ===========================================

void* basic_routine(void* arg) {
    printf("Hello from basic thread\n");
    sleep(1);
    printf("Basic thread ending\n");
    return NULL;
}

// ===========================================
// SECTION 2: Getting return values from threads
// ===========================================

void* roll_dice(void* arg) {
    int value = (rand() % 6) + 1;
    int* result = malloc(sizeof(int));
    *result = value;
    printf("[Dice thread] rolled value %d at address %p\n", *result, result);
    return (void*)result;
}

// ===========================================
// SECTION 3: Passing arguments to threads
// ===========================================

int primes[10] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29};

void* prime_routine(void* arg) {
    int index = *(int*)arg;
    printf("Prime[%d] = %d\n", index, primes[index]);
    free(arg);
    return NULL;
}

// ===========================================
// SECTION 4: Understanding pthread_t and TID
// ===========================================

void* id_routine(void* arg) {
    pthread_t th = pthread_self();
    printf("pthread_t (opaque ID): %lu | Linux TID: %d\n",
           (unsigned long)th, (pid_t)syscall(SYS_gettid));
    return NULL;
}

// ===========================================
// SECTION 5: Detachable threads
// ===========================================

void* detached_routine(void* arg) {
    sleep(1);
    printf("Detached thread finished execution\n");
    return NULL;
}

// ===========================================
// MAIN FUNCTION - Demonstrate all concepts
// ===========================================

int main() {
    srand(time(NULL));

    printf("\n==== 1. Basic thread creation ====\n");
    pthread_t t1, t2;
    pthread_create(&t1, NULL, basic_routine, NULL);
    pthread_create(&t2, NULL, basic_routine, NULL);
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    printf("\n==== 2. Thread returning value ====\n");
    pthread_t dice_thread;
    int* dice_result;
    pthread_create(&dice_thread, NULL, roll_dice, NULL);
    pthread_join(dice_thread, (void**)&dice_result);
    printf("Main thread got dice result = %d (address %p)\n", *dice_result, dice_result);
    free(dice_result);

    printf("\n==== 3. Passing arguments to threads ====\n");
    pthread_t prime_threads[10];
    for (int i = 0; i < 10; i++) {
        int* idx = malloc(sizeof(int));
        *idx = i;
        pthread_create(&prime_threads[i], NULL, prime_routine, idx);
    }
    for (int i = 0; i < 10; i++) {
        pthread_join(prime_threads[i], NULL);
    }

    printf("\n==== 4. pthread_t vs Linux thread ID ====\n");
    pthread_t id_threads[2];
    for (int i = 0; i < 2; i++) {
        pthread_create(&id_threads[i], NULL, id_routine, NULL);
    }
    for (int i = 0; i < 2; i++) {
        pthread_join(id_threads[i], NULL);
    }

    printf("\n==== 5. Detached threads ====\n");
    pthread_t detached[2];
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);

    for (int i = 0; i < 2; i++) {
        pthread_create(&detached[i], &attr, detached_routine, NULL);
    }
    pthread_attr_destroy(&attr);

    printf("Main thread will not join detached threads (they clean up themselves)\n");
    sleep(2);
    printf("All detached threads finished.\n");

    printf("\n==== END OF DEMO ====\n");
    return 0;
}
