#include<stdio.h>
#include "list_utils.c"

struct ListNode* addTwoNumbers(struct ListNode* l1, struct ListNode* l2) {
    int sum = l1->val + l2->val;
    int carry = sum/10;
    struct ListNode* l3 = (struct ListNode*)malloc(sizeof(struct ListNode));
    l3->val = sum%10;
    l3->next = NULL;
    l1 = l1->next;
    l2 = l2->next;
    struct ListNode* nl3 = l3;
    while(l1 || l2 || carry){
        int v1 = 0;
        int v2 = 0;
        if(l1){
            v1 = l1->val;
            l1 = l1->next;
        }
        if(l2){
            v2 = l2->val;
            l2 = l2->next;
        }
        sum = v1 + v2 + carry;
        carry = sum/10;
        nl3->next = (struct ListNode*)malloc(sizeof(struct ListNode));
        nl3->next->val = sum%10;
        nl3->next->next = NULL;
        nl3 = nl3->next;
    }
    return l3;
};


int main() {
    int arr[] = ;
    int size = sizeof(arr) / sizeof(arr[0]);
    addTwoNumbers(arrayToLinkedList({10, 20, 30, 40, 50}));
    printf("Original array: ");
    printArray(arr, size);

    struct ListNode* list = arrayToLinkedList(arr, size);
    printf("Converted to linked list: ");
    printLinkedList(list);

    int newSize;
    int* newArr = linkedListToArray(list, &newSize);
    printf("Converted back to array: ");
    printArray(newArr, newSize);

    free(newArr);
    destroyLinkedList(list);

    return 0;

    int a1 = 2,4,3}
    int a22[] = {5,6,4}
    Output: [7,0,8]
    Explanation: 342 + 465 = 807.
    Example 2:

    Input: l1 = [0], l2 = [0]
    Output: [0]
    Example 3:

    Input: l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
    Output: [8,9,9,9,0,0,0,1]
    
}
