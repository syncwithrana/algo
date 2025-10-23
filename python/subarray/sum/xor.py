def subarrayXor(arr, k):
    res = 0
    pref = {}
    acc = 0

    for val in arr:
        acc ^= val
        res += pref.get(acc ^ k, 0)

        if acc == k:
            res += 1

        pref[acc] = pref.get(acc, 0) + 1

    return res

if __name__ == "__main__":
    arr = [4, 2, 2, 6, 4]
    k = 6

    print(subarrayXor(arr, k))