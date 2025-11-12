#include <stdio.h>
#include <stdlib.h>

struct ListNode {
    int val;
    struct ListNode* next;
};

struct ListNode* createListNode(int val) {
    struct ListNode* newListNode = malloc(sizeof(struct ListNode));
    newListNode->val = val;
    newListNode->next = NULL;
    return newListNode;
}

struct ListNode* arrayToLinkedList(int arr[], int size) {
    if (size == 0) return NULL;

    struct ListNode* head = createListNode(arr[0]);
    struct ListNode* current = head;

    for (int i = 1; i < size; i++) {
        current->next = createListNode(arr[i]);
        current = current->next;
    }

    return head;
}

int* linkedListToArray(struct ListNode* head, int* sizeOut) {
    int count = 0;
    struct ListNode* temp = head;
    while (temp) {
        count++;
        temp = temp->next;
    }

    int* arr = malloc(count * sizeof(int));

    temp = head;
    for (int i = 0; i < count; i++) {
        arr[i] = temp->val;
        temp = temp->next;
    }

    *sizeOut = count;
    return arr;
}

void printLinkedList(struct ListNode* head) {
    while (head) {
        printf("%d -> ", head->val);
        head = head->next;
    }
    printf("NULL\n");
}

void printArray(int arr[], int size) {
    printf("[");
    for (int i = 0; i < size; i++) {
        printf("%d", arr[i]);
        if (i < size - 1) printf(", ");
    }
    printf("]\n");
}

void destroyLinkedList(struct ListNode* list)    {
    struct ListNode* temp;
    while (list) {
        temp = list;
        list = list->next;
        free(temp);
    }
}