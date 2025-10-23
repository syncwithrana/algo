# 1. Subarray Sum Equals K
def subarray_sum_equals_k(arr, k):
    prefix, count = 0, 0
    freq = {0: 1}
    for num in arr:
        prefix += num
        if prefix - k in freq:
            count += freq[prefix - k]
        freq[prefix] = freq.get(prefix, 0) + 1
    return count


# 2. Continuous Subarray Sum (multiple of k)
def continuous_subarray_sum(arr, k):
    prefix = 0
    mp = {0: -1}
    for i, num in enumerate(arr):
        prefix = (prefix + num) % k
        if prefix in mp:
            if i - mp[prefix] > 1:
                return True
        else:
            mp[prefix] = i
    return False


# 3. Find Pivot Index
def find_pivot_index(arr):
    total = sum(arr)
    prefix = 0
    for i in range(len(arr)):
        if prefix == total - prefix - arr[i]:
            return i
        prefix += arr[i]
    return -1


# 4. Subarray Sums Divisible by K
def subarrays_div_by_k(arr, k):
    prefix, count = 0, 0
    freq = {0: 1}
    for num in arr:
        prefix = (prefix + num) % k
        if prefix < 0:
            prefix += k
        count += freq.get(prefix, 0)
        freq[prefix] = freq.get(prefix, 0) + 1
    return count


# 5. Binary Subarrays With Sum
def binary_subarrays_with_sum(arr, goal):
    def atMost(G):
        if G < 0: return 0
        left, curr, count = 0, 0, 0
        for right in range(len(arr)):
            curr += arr[right]
            while curr > G:
                curr -= arr[left]; left += 1
            count += right - left + 1
        return count
    return atMost(goal) - atMost(goal - 1)


# 6. Count Number of Nice Subarrays
def count_nice_subarrays(arr, k):
    arr = [x % 2 for x in arr]
    return binary_subarrays_with_sum(arr, k)


# 7. Maximum Size Subarray Sum = K
def max_size_subarray_sum_k(arr, k):
    prefix, maxlen = 0, 0
    mp = {}
    for i, num in enumerate(arr):
        prefix += num
        if prefix == k:
            maxlen = i + 1
        if (prefix - k) in mp:
            maxlen = max(maxlen, i - mp[prefix - k])
        if prefix not in mp:
            mp[prefix] = i
    return maxlen


# 8. Minimum Size Subarray Sum ≥ target
def min_size_subarray_sum(arr, target):
    left, total, res = 0, 0, float('inf')
    for right in range(len(arr)):
        total += arr[right]
        while total >= target:
            res = min(res, right - left + 1)
            total -= arr[left]
            left += 1
    return 0 if res == float('inf') else res


# 9. Longest Well-Performing Interval
def longest_well_performing_interval(hours):
    score, mp, res = 0, {}, 0
    for i, h in enumerate(hours):
        score += 1 if h > 8 else -1
        if score > 0:
            res = i + 1
        elif (score - 1) in mp:
            res = max(res, i - mp[score - 1])
        if score not in mp:
            mp[score] = i
    return res


# 10. Range Sum Query (Immutable)
def build_prefix_sum(arr):
    prefix = [0]
    for num in arr:
        prefix.append(prefix[-1] + num)
    return prefix

def range_sum(prefix, l, r):
    return prefix[r+1] - prefix[l]


# 11. Matrix Block Sum (2D Prefix)
def matrix_block_sum(mat, k):
    n, m = len(mat), len(mat[0])
    P = [[0]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(1, m+1):
            P[i][j] = mat[i-1][j-1] + P[i-1][j] + P[i][j-1] - P[i-1][j-1]
    ans = [[0]*m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            x1, y1 = max(0, i-k), max(0, j-k)
            x2, y2 = min(n-1, i+k), min(m-1, j+k)
            ans[i][j] = P[x2+1][y2+1] - P[x1][y2+1] - P[x2+1][y1] + P[x1][y1]
    return ans


# 12. Number of Submatrices That Sum to Target
def num_submat_sum_target(matrix, target):
    rows, cols = len(matrix), len(matrix[0])
    count = 0
    for top in range(rows):
        arr = [0]*cols
        for bottom in range(top, rows):
            for c in range(cols):
                arr[c] += matrix[bottom][c]
            prefix, freq = 0, {0:1}
            for num in arr:
                prefix += num
                if prefix - target in freq:
                    count += freq[prefix - target]
                freq[prefix] = freq.get(prefix, 0) + 1
    return count


# 13. Ways to Make a Fair Array
def ways_to_make_fair(arr):
    n = len(arr)
    pre_even, pre_odd = [0]* (n+1), [0]* (n+1)
    for i in range(n):
        pre_even[i+1] = pre_even[i]
        pre_odd[i+1] = pre_odd[i]
        if i % 2 == 0:
            pre_even[i+1] += arr[i]
        else:
            pre_odd[i+1] += arr[i]
    count = 0
    for i in range(n):
        left_even, left_odd = pre_even[i], pre_odd[i]
        right_even = pre_odd[n] - pre_odd[i+1]
        right_odd = pre_even[n] - pre_even[i+1]
        if left_even + right_even == left_odd + right_odd:
            count += 1
    return count


# 14. Max Sum of Two Non-Overlapping Subarrays
def max_sum_two_no_overlap(arr, L, M):
    def max_sum(L, M):
        prefix = [0]
        for x in arr:
            prefix.append(prefix[-1] + x)
        res, Lmax = 0, 0
        for i in range(L+M, len(prefix)):
            Lmax = max(Lmax, prefix[i-M] - prefix[i-M-L])
            res = max(res, Lmax + prefix[i] - prefix[i-M])
        return res
    return max(max_sum(L, M), max_sum(M, L))


# 15. Maximum Average Subarray (Fixed length k)
def max_average_subarray(arr, k):
    prefix = [0]
    for num in arr:
        prefix.append(prefix[-1] + num)
    max_avg = float('-inf')
    for i in range(k, len(prefix)):
        avg = (prefix[i] - prefix[i-k]) / k
        max_avg = max(max_avg, avg)
    return max_avg


# ---------- DRIVER CODE ----------
if __name__ == "__main__":
    arr = list(map(int, input("Enter array elements: ").split()))
    k = int(input("Enter k / target value: "))

    print("Subarray Sum Equals K:", subarray_sum_equals_k(arr, k))
    print("Continuous Subarray Sum Multiple of K:", continuous_subarray_sum(arr, k))
    print("Pivot Index:", find_pivot_index(arr))
    print("Subarrays Divisible by K:", subarrays_div_by_k(arr, k))
    print("Count Nice Subarrays (odd-count=k):", count_nice_subarrays(arr, k))
    print("Max Size Subarray Sum = K:", max_size_subarray_sum_k(arr, k))
    print("Min Size Subarray Sum ≥ target:", min_size_subarray_sum(arr, k))
    print("Max Average Subarray (len=k):", max_average_subarray(arr, k))
