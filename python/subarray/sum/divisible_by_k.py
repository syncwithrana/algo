def subCount(arr, k):
    res = 0
    curr_mod = 0
    pref_count = {}

    for item in arr:
        curr_mod = ((curr_mod + item) % k + k) % k

        if curr_mod == 0:
            res += 1

        if curr_mod in pref_count:
            res += pref_count[curr_mod]

        pref_count[curr_mod] = pref_count.get(curr_mod, 0) + 1

    return res

if __name__ == "__main__":
    arr = [4, 5, 0, -2, -3, 1]
    k = 5
    print(subCount(arr, k))
