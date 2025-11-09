int subarraysDivByK(int* nums, int numsSize, int k) {
    for(int i = 0; i < numsSize; i++)   {
        pref_sum = (pref_sum + nums[i]) % k;
        hash_type* item = hash_get_item(pref_hash, pref_sum % k);
        if(item1)   {
            res += item->val;
        }
        else {
            hash_add_item(pref_hash, item->val, 1);
        }
    }
    hash_delete_all(&item);
    return res;
}Sum