def cntSubarrays(arr, k):
    pref_count = {}
    res = 0
    curr_sum = 0

    for item in arr:
        curr_sum += item

        if curr_sum == k:
            res += 1

        if curr_sum - k in pref_count:
            res += pref_count[curr_sum - k]

        pref_count[curr_sum] = pref_count.get(curr_sum, 0) + 1

    return res

if __name__ == "__main__":
    arr = [10, 2, -2, -20, 10]
    k = -10
    print(cntSubarrays(arr, k))