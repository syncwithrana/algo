1 Subarray Sum Equals K (Prefix Sum + Hash Map)
Count subarrays with sum = K.
```
def subarray_sum(nums, k):
    count = total = 0
    pref = {0: 1}
    for x in nums:
        total += x
        if total - k in pref:
            count += pref[total - k]
        pref[total] = pref.get(total, 0) + 1
    return count
```

2 Subarray Sum Divisible by K
```
def subarrays_div_by_k(nums, k):
    pref, count, total = {0: 1}, 0, 0
    for x in nums:
        total = (total + x) % k
        count += pref.get(total, 0)
        pref[total] = pref.get(total, 0) + 1
    return count
```

3 Longest Subarray with Sum K
```
def longest_subarray_sum_k(nums, k):
    pref, total, ans = {0: -1}, 0, 0
    for i, x in enumerate(nums):
        total += x
        if total - k in pref:
            ans = max(ans, i - pref[total - k])
        if total not in pref:
            pref[total] = i
    return ans
```

4 Maximum Product Subarray
```
def max_product(nums):
    cur_max = cur_min = ans = nums[0]
    for x in nums[1:]:
        if x < 0: cur_max, cur_min = cur_min, cur_max
        cur_max = max(x, cur_max * x)
        cur_min = min(x, cur_min * x)
        ans = max(ans, cur_max)
    return ans
```

5 Count Subarrays with Equal 0s and 1s
Treat 0 as -1 → use prefix-sum hashmap.
```
def count_equal_0_1(nums):
    pref, total, ans = {0: 1}, 0, 0
    for x in nums:
        total += 1 if x == 1 else -1
        ans += pref.get(total, 0)
        pref[total] = pref.get(total, 0) + 1
    return ans
```

6 Maximum Length of Contiguous 1s (After Flipping at Most K 0s)
Sliding window.
```
def longest_ones(nums, k):
    left = 0
    for right in range(len(nums)):
        if nums[right] == 0:
            k -= 1
        if k < 0:
            if nums[left] == 0:
                k += 1
            left += 1
    return right - left + 1
```

7 Longest Subarray with Equal Number of 0, 1, and 2
Use tuple of prefix differences.
```
def longest_equal_012(nums):
    count0 = count1 = count2 = ans = 0
    seen = {(0, 0): -1}
    for i, x in enumerate(nums):
        if x == 0: count0 += 1
        elif x == 1: count1 += 1
        else: count2 += 1
        diff1, diff2 = count1 - count0, count2 - count1
        if (diff1, diff2) in seen:
            ans = max(ans, i - seen[(diff1, diff2)])
        else:
            seen[(diff1, diff2)] = i
    return ans
```

8 Minimum Size Subarray Sum ≥ Target
Sliding window (O(n)).
```
def min_subarray_len(target, nums):
    left = total = 0
    res = float('inf')
    for right in range(len(nums)):
        total += nums[right]
        while total >= target:
            res = min(res, right - left + 1)
            total -= nums[left]
            left += 1
    return res if res != float('inf') else 0
```

9 Longest Subarray with Positive Product
```
def get_max_len(nums):
    pos = neg = ans = 0
    for x in nums:
        if x == 0:
            pos = neg = 0
        elif x > 0:
            pos += 1
            neg = neg + 1 if neg else 0
        else:
            pos, neg = neg + 1 if neg else 0, pos + 1
        ans = max(ans, pos)
    return ans
```

10 Count Subarrays with Given XOR
```
def subarray_xor(nums, k):
    pref, total, ans = {0: 1}, 0, 0
    for x in nums:
        total ^= x
        ans += pref.get(total ^ k, 0)
        pref[total] = pref.get(total, 0) + 1
    return ans
```

11 Sort Colors (Dutch National Flag Problem)
```
In-place sort of 0s, 1s, 2s.
def sort_colors(nums):
    low = mid = 0
    high = len(nums) - 1
    while mid <= high:
        if nums[mid] == 0:
            nums[low], nums[mid] = nums[mid], nums[low]
            low += 1; mid += 1
        elif nums[mid] == 1:
            mid += 1
        else:
            nums[mid], nums[high] = nums[high], nums[mid]
            high -= 1
    return nums
```

12 Merge Intervals
```
def merge(intervals):
    intervals.sort(key=lambda x: x[0])
    res = [intervals[0]]
    for start, end in intervals[1:]:
        if start <= res[-1][1]:
            res[-1][1] = max(res[-1][1], end)
        else:
            res.append([start, end])
    return res
```

13 3Sum Problem
Find all unique triplets summing to 0.
```
def three_sum(nums):
    nums.sort()
    res = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]: continue
        l, r = i + 1, len(nums) - 1
        while l < r:
            s = nums[i] + nums[l] + nums[r]
            if s < 0: l += 1
            elif s > 0: r -= 1
            else:
                res.append([nums[i], nums[l], nums[r]])
                l += 1; r -= 1
                while l < r and nums[l] == nums[l - 1]: l += 1
                while l < r and nums[r] == nums[r + 1]: r -= 1
    return res
```

14 4Sum
Same idea as 3Sum, with an extra layer.
```
def four_sum(nums, target):
    nums.sort()
    res = []
    n = len(nums)
    for i in range(n - 3):
        if i > 0 and nums[i] == nums[i - 1]: continue
        for j in range(i + 1, n - 2):
            if j > i + 1 and nums[j] == nums[j - 1]: continue
            l, r = j + 1, n - 1
            while l < r:
                s = nums[i] + nums[j] + nums[l] + nums[r]
                if s == target:
                    res.append([nums[i], nums[j], nums[l], nums[r]])
                    l += 1; r -= 1
                    while l < r and nums[l] == nums[l - 1]: l += 1
                    while l < r and nums[r] == nums[r + 1]: r -= 1
                elif s < target: l += 1
                else: r -= 1
    return res
```

15 Trapping Rain Water
Classic elevation map problem.
```
def trap(height):
    n = len(height)
    if n == 0: return 0
    left, right = [0]*n, [0]*n
    left[0] = height[0]
    for i in range(1, n):
        left[i] = max(left[i-1], height[i])
    right[-1] = height[-1]
    for i in range(n-2, -1, -1):
        right[i] = max(right[i+1], height[i])
    return sum(min(left[i], right[i]) - height[i] for i in range(n))
```

16 Next Permutation
Rearrange numbers to next lexicographical permutation.
```
def next_permutation(nums):
    i = len(nums) - 2
    while i >= 0 and nums[i] >= nums[i+1]:
        i -= 1
    if i >= 0:
        j = len(nums) - 1
        while nums[j] <= nums[i]:
            j -= 1
        nums[i], nums[j] = nums[j], nums[i]
    nums[i+1:] = reversed(nums[i+1:])
    return nums
```

17 Majority Element (> n/2 times)
Boyer-Moore Voting Algorithm.
```
def majority_element(nums):
    count = candidate = 0
    for x in nums:
        if count == 0: candidate = x
        count += (1 if x == candidate else -1)
    return candidate
```

18 Rearrange Array Alternating + and - (Maintain Order)
```
def rearrange_pos_neg(nums):
    pos = [x for x in nums if x >= 0]
    neg = [x for x in nums if x < 0]
    res = []
    i = j = 0
    while i < len(pos) and j < len(neg):
        res.append(pos[i]); res.append(neg[j])
        i += 1; j += 1
    res.extend(pos[i:]); res.extend(neg[j:])
    return res
```

19 Product of Array Except Self
No division, O(n).
```
def product_except_self(nums):
    n = len(nums)
    res = [1]*n
    prefix = 1
    for i in range(n):
        res[i] = prefix
        prefix *= nums[i]
    suffix = 1
    for i in range(n-1, -1, -1):
        res[i] *= suffix
        suffix *= nums[i]
    return res
```

20 Minimum Swaps to Sort Array
Use cycles detection.
```
def min_swaps(nums):
    arr = list(enumerate(nums))
    arr.sort(key=lambda x: x[1])
    visited = [False]*len(nums)
    swaps = 0
    for i in range(len(nums)):
        if visited[i] or arr[i][0] == i:
            continue
        cycle = 0
        j = i
        while not visited[j]:
            visited[j] = True
            j = arr[j][0]
            cycle += 1
        swaps += cycle - 1
    return swaps
```

21 Classic Maximum Subarray (Kadane)
Return max contiguous subarray sum.
```
def max_subarray(nums):
    cur = best = nums[0]
    for x in nums[1:]:
        cur = max(x, cur + x)
        best = max(best, cur)
    return best
```

22 Maximum Circular Subarray Sum (Kadane + Inversion)
Handle wrap-around.
```
def max_subarray_circular(nums):
    # standard Kadane
    def kadane(arr):
        cur = best = arr[0]
        for x in arr[1:]:
            cur = max(x, cur + x)
            best = max(best, cur)
        return best

    max_kadane = kadane(nums)
    total = sum(nums)
    # Kadane on inverted array gives min subarray sum
    min_kadane = -kadane([-x for x in nums])
    # If all numbers are negative, max_kadane is the answer
    return max_kadane if min_kadane == total else max(max_kadane, total - min_kadane)
```

23 Sliding Window Maximum (Deque / Monotonic Queue)
O(n) max for each window.
```
from collections import deque
def max_sliding_window(nums, k):
    dq = deque()  # store indices, decreasing values
    res = []
    for i, x in enumerate(nums):
        while dq and dq[0] <= i - k:
            dq.popleft()
        while dq and nums[dq[-1]] < x:
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            res.append(nums[dq[0]])
    return res
```

24 Next Greater Element (Monotonic Stack)
Return next greater for each element (or -1).
```
def next_greater(nums):
    res = [-1] * len(nums)
    stack = []
    for i, x in enumerate(nums):
        while stack and nums[stack[-1]] < x:
            res[stack.pop()] = x
        stack.append(i)
    return res
```

25 Largest Rectangle in Histogram (Monotonic Stack)
Classic area-in-histogram.
```
def largest_rectangle_area(heights):
    stack = []
    max_area = 0
    for i, h in enumerate(heights + [0]):
        while stack and heights[stack[-1]] > h:
            H = heights[stack.pop()]
            L = stack[-1] + 1 if stack else 0
            max_area = max(max_area, H * (i - L))
        stack.append(i)
    return max_area
```

26 Max Sum Rectangle in 2D Matrix (Kadane 2D)
Reduce columns and run Kadane on rows — O(cols^2 * rows).
```
def max_sum_rectangle(matrix):
    if not matrix or not matrix[0]:
        return 0
    rows, cols = len(matrix), len(matrix[0])
    best = float('-inf')
    for left in range(cols):
        row_sum = [0] * rows
        for right in range(left, cols):
            for r in range(rows):
                row_sum[r] += matrix[r][right]
            # Kadane on row_sum
            cur = row_sum[0]
            best = max(best, cur)
            for v in row_sum[1:]:
                cur = max(v, cur + v)
                best = max(best, cur)
    return best
```

27 Sum of Subarray Minimums (Stack + Contribution)
Compute sum of min over all subarrays mod large prime if needed.
```
def sum_subarray_mins(arr):
    MOD = 10**9 + 7
    n = len(arr)
    stack = []
    left = [0] * n
    for i in range(n):
        count = 1
        while stack and stack[-1][0] > arr[i]:
            count += stack.pop()[1]
        stack.append((arr[i], count))
        left[i] = count

    stack.clear()
    right = [0] * n
    for i in range(n-1, -1, -1):
        count = 1
        while stack and stack[-1][0] >= arr[i]:
            count += stack.pop()[1]
        stack.append((arr[i], count))
        right[i] = count

    return sum(arr[i] * left[i] * right[i] for i in range(n)) % MOD
```

28 Maximum Profit with At Most k Transactions (DP / Optimized)
O(nk) dynamic programming.
```
def max_profit_k(prices, k):
    if not prices: return 0
    n = len(prices)
    if k >= n // 2:
        # unlimited transactions
        return sum(max(0, prices[i+1] - prices[i]) for i in range(n - 1))
    dp = [[0] * n for _ in range(k + 1)]
    for t in range(1, k + 1):
        best = -prices[0]
        for i in range(1, n):
            dp[t][i] = max(dp[t][i-1], prices[i] + best)
            best = max(best, dp[t-1][i] - prices[i])
    return dp[k][n-1]
```

29 Best Time to Buy and Sell Stock with Cooldown
DP with O(n) time and O(1) space.
```
def max_profit_with_cooldown(prices):
    if not prices: return 0
    buy = -prices[0]
    sell = 0
    prev_sell = 0
    for p in prices[1:]:
        prev_buy = buy
        buy = max(buy, prev_sell - p)
        prev_sell = sell
        sell = max(sell, prev_buy + p)
    return sell
```

30 Sum of Subarray Ranges (max - min for each subarray)
Use contribution technique with monotonic stacks (two passes).
```
def subarray_ranges(nums):
    n = len(nums)
    # contribution for max
    stack = []
    left_max = [0]*n
    for i in range(n):
        cnt = 1
        while stack and nums[stack[-1][0]] < nums[i]:
            cnt += stack.pop()[1]
        stack.append((i, cnt))
        left_max[i] = cnt
    stack.clear()
    right_max = [0]*n
    for i in range(n-1, -1, -1):
        cnt = 1
        while stack and nums[stack[-1][0]] <= nums[i]:
            cnt += stack.pop()[1]
        stack.append((i, cnt))
        right_max[i] = cnt

    # contribution for min
    stack.clear()
    left_min = [0]*n
    for i in range(n):
        cnt = 1
        while stack and nums[stack[-1][0]] > nums[i]:
            cnt += stack.pop()[1]
        stack.append((i, cnt))
        left_min[i] = cnt
    stack.clear()
    right_min = [0]*n
    for i in range(n-1, -1, -1):
        cnt = 1
        while stack and nums[stack[-1][0]] >= nums[i]:
            cnt += stack.pop()[1]
        stack.append((i, cnt))
        right_min[i] = cnt

    total = 0
    for i in range(n):
        total += nums[i] * (left_max[i] * right_max[i] - left_min[i] * right_min[i])
    return total
```

31 Jump Game I — Can Reach End
Greedy reachability check.
```
def can_jump(nums):
    reach = 0
    for i, x in enumerate(nums):
        if i > reach: return False
        reach = max(reach, i + x)
    return True
```

32 Jump Game II — Minimum Jumps
Greedy layer-by-layer jump.
```
def jump(nums):
    jumps = cur_end = cur_far = 0
    for i in range(len(nums) - 1):
        cur_far = max(cur_far, i + nums[i])
        if i == cur_end:
            jumps += 1
            cur_end = cur_far
    return jumps
```

33 Partition Equal Subset Sum
DP subset-sum variant.
```
def can_partition(nums):
    total = sum(nums)
    if total % 2: return False
    target = total // 2
    dp = {0}
    for x in nums:
        dp |= {x + v for v in dp if x + v <= target}
    return target in dp
```

34 Minimum Moves to Equal Array Elements II
Median minimizes moves.
```
def min_moves2(nums):
    nums.sort()
    median = nums[len(nums)//2]
    return sum(abs(x - median) for x in nums)
```


35 Maximum Product of Three Numbers
```
def maximum_product(nums):
    nums.sort()
    return max(nums[-1]*nums[-2]*nums[-3], nums[0]*nums[1]*nums[-1])
```

36 Rearrange Array by Parity and Index
Even indices → even numbers, odd indices → odd numbers.
```
def sort_array_by_parity_ii(nums):
    even, odd = 0, 1
    n = len(nums)
    res = [0]*n
    for x in nums:
        if x % 2 == 0:
            res[even] = x; even += 2
        else:
            res[odd] = x; odd += 2
    return res
```

37 Find the Duplicate Number (Floyd’s Cycle Detection)
Detect cycle in index graph.
```
def find_duplicate(nums):
    slow = fast = nums[0]
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast: break
    slow = nums[0]
    while slow != fast:
        slow = nums[slow]
        fast = nums[fast]
    return slow
```

38 Missing and Repeating Number (Math / XOR)
Given 1..n with one duplicate and one missing.
```
def find_missing_and_repeating(nums):
    n = len(nums)
    xor_all = 0
    for i in range(1, n+1):
        xor_all ^= i ^ nums[i-1]
    set_bit = xor_all & -xor_all
    a = b = 0
    for i in range(n):
        if nums[i] & set_bit: a ^= nums[i]
        else: b ^= nums[i]
        if (i+1) & set_bit: a ^= (i+1)
        else: b ^= (i+1)
    # verify which is missing
    if a in nums:
        return (a, b)  # (repeat, missing)
    else:
        return (b, a)
```

39 Split Array Largest Sum (Binary Search on Answer)
Minimize largest subarray sum after splitting into k parts.
```
def split_array(nums, k):
    def can(mid):
        parts, cur = 1, 0
        for x in nums:
            if cur + x > mid:
                parts += 1
                cur = 0
            cur += x
        return parts <= k
    l, r = max(nums), sum(nums)
    while l < r:
        mid = (l + r) // 2
        if can(mid): r = mid
        else: l = mid + 1
    return l
```

40 Allocate Minimum Number of Pages (Same Logic)
Classic variant of splitting workloads.
```
def allocate_pages(pages, students):
    def feasible(limit):
        total = cnt = 0
        for p in pages:
            if p > limit: return False
            total += p
            if total > limit:
                cnt += 1
                total = p
        cnt += 1
        return cnt <= students
    l, r = max(pages), sum(pages)
    ans = r
    while l <= r:
        mid = (l + r)//2
        if feasible(mid):
            ans = mid
            r = mid - 1
        else:
            l = mid + 1
    return ans
```

41 Range Addition (Difference Array)
Apply many range-add updates quickly; then produce final array.
```
def range_add(n, updates):
    # n = length, updates = [(l, r, val), ...] inclusive indices
    diff = [0] * (n + 1)
    for l, r, val in updates:
        diff[l] += val
        diff[r + 1] -= val
    res = [0] * n
    cur = 0
    for i in range(n):
        cur += diff[i]
        res[i] = cur
    return res
```
Note: O(1) per update, O(n) to finalize.


42 Interval Scheduling — Maximum Number of Non-overlapping Intervals
Greedy by finish time.
```
def max_non_overlapping(intervals):
    if not intervals: return 0
    intervals.sort(key=lambda x: x[1])
    cnt = 1
    end = intervals[0][1]
    for s, e in intervals[1:]:
        if s >= end:
            cnt += 1
            end = e
    return cnt
```

43 Meeting Rooms II — Minimum Number of Rooms (Min-heap)
Find peak concurrent meetings.
```
import heapq
def min_meeting_rooms(intervals):
    if not intervals: return 0
    intervals.sort(key=lambda x: x[0])
    heap = []  # end times
    heapq.heappush(heap, intervals[0][1])
    for s, e in intervals[1:]:
        if heap[0] <= s:
            heapq.heappop(heap)
        heapq.heappush(heap, e)
    return len(heap)
```
Alternative sweep-line also possible (sort events).


44 Maximum Overlapping Intervals (Sweep Line)
Return maximum number of intervals overlapping at any point.
```
def max_overlap(intervals):
    events = []
    for s, e in intervals:
        events.append((s, 1))
        events.append((e, -1))
    events.sort(key=lambda x: (x[0], -x[1]))  # start before end at same time
    cur = best = 0
    for _, delta in events:
        cur += delta
        best = max(best, cur)
    return best
```

45 Longest Mountain in Array
Strictly increasing then strictly decreasing.
```
def longest_mountain(arr):
    n = len(arr)
    if n < 3: return 0
    up = [0]*n
    down = [0]*n
    for i in range(1, n):
        if arr[i] > arr[i-1]:
            up[i] = up[i-1] + 1
    for i in range(n-2, -1, -1):
        if arr[i] > arr[i+1]:
            down[i] = down[i+1] + 1
    best = 0
    for i in range(n):
        if up[i] and down[i]:
            best = max(best, up[i] + down[i] + 1)
    return best
```

46 Longest Turbulent Subarray
Alternating comparisons (> < > ...).
```
def max_turbulence_size(arr):
    n = len(arr)
    if n < 2: return n
    up = down = best = 1
    for i in range(1, n):
        if arr[i] > arr[i-1]:
            up = down + 1
            down = 1
        elif arr[i] < arr[i-1]:
            down = up + 1
            up = 1
        else:
            up = down = 1
        best = max(best, up, down)
    return best
```

47 Count Inversions (Modified Merge Sort)
Number of pairs i < j with arr[i] > arr[j].
```
def count_inversions(arr):
    def merge_count(a):
        n = len(a)
        if n <= 1: return a, 0
        mid = n//2
        left, lc = merge_count(a[:mid])
        right, rc = merge_count(a[mid:])
        merged = []
        i = j = inv = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i]); i += 1
            else:
                merged.append(right[j]); j += 1
                inv += len(left) - i
        merged.extend(left[i:]); merged.extend(right[j:])
        return merged, inv + lc + rc
    _, ans = merge_count(arr)
    return ans
```
O(n log n).


48 Range Minimum Query (RMQ) — Sparse Table (O(1) query, O(n log n) build)
Static array RMQ (idempotent function: min).
```
import math
class SparseTableRMQ:
    def __init__(self, arr):
        self.n = len(arr)
        self.LOG = math.floor(math.log2(self.n)) + 1
        self.st = [[0]*self.n for _ in range(self.LOG)]
        self.st[0] = arr[:]
        j = 1
        while (1 << j) <= self.n:
            length = 1 << j
            half = length >> 1
            for i in range(self.n - length + 1):
                self.st[j][i] = min(self.st[j-1][i], self.st[j-1][i+half])
            j += 1

    def query(self, l, r):
        # inclusive l,r
        k = (r - l + 1).bit_length() - 1
        return min(self.st[k][l], self.st[k][r - (1<<k) + 1])
```
Good for many static queries.


49 Smallest Subarray Containing All Distinct Elements of the Array
Find minimal-length subarray that contains every unique value at least once.
```
def smallest_subarray_all_distinct(arr):
    need = len(set(arr))
    freq = {}
    l = 0
    best = float('inf')
    have = 0
    for r, x in enumerate(arr):
        freq[x] = freq.get(x, 0) + 1
        if freq[x] == 1:
            have += 1
        while have == need:
            best = min(best, r - l + 1)
            freq[arr[l]] -= 1
            if freq[arr[l]] == 0:
                have -= 1
            l += 1
    return best if best != float('inf') else 0
```

50 K Smallest Pairs with Smallest Sums (Heap)
Return k pairs (nums1[i], nums2[j]) with smallest sums.
```
import heapq
def k_smallest_pairs(nums1, nums2, k):
    if not nums1 or not nums2 or k == 0: return []
    n1, n2 = len(nums1), len(nums2)
    heap = [(nums1[0] + nums2[0], 0, 0)]
    seen = {(0,0)}
    res = []
    while heap and len(res) < k:
        s, i, j = heapq.heappop(heap)
        res.append((nums1[i], nums2[j]))
        if i + 1 < n1 and (i+1, j) not in seen:
            heapq.heappush(heap, (nums1[i+1] + nums2[j], i+1, j))
            seen.add((i+1, j))
        if j + 1 < n2 and (i, j+1) not in seen:
            heapq.heappush(heap, (nums1[i] + nums2[j+1], i, j+1))
            seen.add((i, j+1))
    return res
```

51 Longest Subarray with at Most K Distinct Elements
Classic sliding window.
```
def longest_k_distinct(arr, k):
    freq, left, best = {}, 0, 0
    for right, val in enumerate(arr):
        freq[val] = freq.get(val, 0) + 1
        while len(freq) > k:
            freq[arr[left]] -= 1
            if freq[arr[left]] == 0:
                del freq[arr[left]]
            left += 1
        best = max(best, right - left + 1)
    return best
```
✅ Common pattern for “at most k” substring/subarray problems.


52 Longest Subarray with Sum ≤ K
Monotonic prefix deque technique.
```
from collections import deque
def longest_subarray_sum_leq_k(arr, K):
    n = len(arr)
    pref = [0]
    for x in arr:
        pref.append(pref[-1] + x)
    dq = deque()
    best = 0
    for i, val in enumerate(pref):
        while dq and val - pref[dq[0]] > K:
            dq.popleft()
        if dq:
            best = max(best, i - dq[0])
        while dq and pref[dq[-1]] >= val:
            dq.pop()
        dq.append(i)
    return best
```
Tricky but elegant.


53 Longest Subarray with Equal 0s and 1s
Convert 0 → -1, find longest subarray with sum = 0.
```
def longest_equal_0_1(arr):
    mp = {0: -1}
    s = 0
    best = 0
    for i, x in enumerate(arr):
        s += 1 if x == 1 else -1
        if s in mp:
            best = max(best, i - mp[s])
        else:
            mp[s] = i
    return best
```

54 Longest Subarray with Equal Number of 0s, 1s, and 2s
Use tuple of counts difference as key.
```
def longest_equal_0_1_2(arr):
    count0 = count1 = count2 = 0
    mp = {(0,0): -1}
    best = 0
    for i, x in enumerate(arr):
        if x == 0: count0 += 1
        elif x == 1: count1 += 1
        else: count2 += 1
        key = (count1 - count0, count2 - count1)
        if key in mp:
            best = max(best, i - mp[key])
        else:
            mp[key] = i
    return best
```
Elegant counting trick.


55 Partition Array into 3 Parts with Equal Sum
Return True/False.
```
def can_three_parts_equal_sum(arr):
    total = sum(arr)
    if total % 3 != 0: return False
    target = total // 3
    cur, count = 0, 0
    for x in arr:
        cur += x
        if cur == target:
            count += 1
            cur = 0
    return count >= 3
```

56 Shortest Subarray with Sum ≥ K
Monotonic deque on prefix sums.
```
from collections import deque
def shortest_subarray_sum_ge_k(arr, K):
    n = len(arr)
    pref = [0]
    for x in arr: pref.append(pref[-1] + x)
    dq = deque()
    best = float('inf')
    for i, cur in enumerate(pref):
        while dq and cur - pref[dq[0]] >= K:
            best = min(best, i - dq.popleft())
        while dq and pref[dq[-1]] >= cur:
            dq.pop()
        dq.append(i)
    return best if best != float('inf') else -1
```
Famous hard-level Leetcode pattern.


57 Count Subarrays with Product Less Than K
Sliding window using multiplication.
```
def num_subarray_product_less_than_k(arr, k):
    if k <= 1: return 0
    prod = 1
    left = count = 0
    for right, val in enumerate(arr):
        prod *= val
        while prod >= k:
            prod //= arr[left]
            left += 1
        count += right - left + 1
    return count
```
Sliding window for multiplicative property.


58 Subarray Bitwise ORs
Find distinct bitwise OR values of all subarrays.
```
def subarray_bitwise_ors(arr):
    cur, res = set(), set()
    for x in arr:
        cur = {x | y for y in cur} | {x}
        res |= cur
    return len(res)
```
🧠 Trick: Maintain current OR set (monotonic).


59 Maximum XOR of Two Numbers in Array
Use Trie (bitwise 0/1 path).
```
class TrieNode:
    def __init__(self):
        self.children = {}

def max_xor(arr):
    root = TrieNode()
    for num in arr:
        node = root
        for i in range(31, -1, -1):
            b = (num >> i) & 1
            if b not in node.children:
                node.children[b] = TrieNode()
            node = node.children[b]
    best = 0
    for num in arr:
        node = root
        cur = 0
        for i in range(31, -1, -1):
            b = (num >> i) & 1
            if 1 - b in node.children:
                cur |= 1 << i
                node = node.children[1 - b]
            else:
                node = node.children[b]
        best = max(best, cur)
    return best
```
Classic bitwise-trie question.


60 Count Subarrays with XOR = K
Hash prefix XORs.
```
def count_subarrays_xor_k(arr, k):
    freq = {0: 1}
    xorsum = 0
    count = 0
    for x in arr:
        xorsum ^= x
        count += freq.get(xorsum ^ k, 0)
        freq[xorsum] = freq.get(xorsum, 0) + 1
    return count
```

61 Find Peak Element (Binary Search)
Find any index i where arr[i] > arr[i±1].
```
def find_peak(arr):
    l, r = 0, len(arr) - 1
    while l < r:
        m = (l + r) // 2
        if arr[m] < arr[m + 1]:
            l = m + 1
        else:
            r = m
    return l
```
O(log n) binary search on pattern direction.


62 Find Minimum in Rotated Sorted Array
```
def find_min_rotated(arr):
    l, r = 0, len(arr) - 1
    while l < r:
        m = (l + r) // 2
        if arr[m] > arr[r]:
            l = m + 1
        else:
            r = m
    return arr[l]
```
Handles rotation edge cases cleanly.


63 Search in Rotated Sorted Array
Binary search with condition halves.
```
def search_rotated(arr, target):
    l, r = 0, len(arr) - 1
    while l <= r:
        m = (l + r) // 2
        if arr[m] == target:
            return m
        if arr[l] <= arr[m]:
            if arr[l] <= target < arr[m]:
                r = m - 1
            else:
                l = m + 1
        else:
            if arr[m] < target <= arr[r]:
                l = m + 1
            else:
                r = m - 1
    return -1
```

64 K-th Smallest Pair Distance
Binary search on distance + sliding window count.
```
def kth_smallest_pair_distance(arr, k):
    arr.sort()
    def count_leq(dist):
        cnt = 0
        l = 0
        for r, val in enumerate(arr):
            while val - arr[l] > dist:
                l += 1
            cnt += r - l
        return cnt
    lo, hi = 0, arr[-1] - arr[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if count_leq(mid) >= k:
            hi = mid
        else:
            lo = mid + 1
    return lo
```

65 Median of Two Sorted Arrays
Binary search partition approach.
```
def median_two_sorted(a, b):
    if len(a) > len(b): a, b = b, a
    n, m = len(a), len(b)
    l, r = 0, n
    half = (n + m + 1) // 2
    while l <= r:
        i = (l + r) // 2
        j = half - i
        Aleft = a[i-1] if i > 0 else float('-inf')
        Aright = a[i] if i < n else float('inf')
        Bleft = b[j-1] if j > 0 else float('-inf')
        Bright = b[j] if j < m else float('inf')
        if Aleft <= Bright and Bleft <= Aright:
            if (n + m) % 2:
                return max(Aleft, Bleft)
            return (max(Aleft, Bleft) + min(Aright, Bright)) / 2
        elif Aleft > Bright:
            r = i - 1
        else:
            l = i + 1
🧩 Hard classic — O(log (min n, m)).
```

66 Sum of Subarray Minimums (Monotonic Stack)
Each element contributes as min in some subarrays.
```
def sum_subarray_mins(arr):
    n = len(arr)
    mod = 10**9 + 7
    left, right = [0]*n, [0]*n
    stack = []
    for i, x in enumerate(arr):
        count = 1
        while stack and stack[-1][0] > x:
            count += stack.pop()[1]
        left[i] = count
        stack.append((x, count))
    stack.clear()
    for i in range(n-1, -1, -1):
        count = 1
        while stack and stack[-1][0] >= arr[i]:
            count += stack.pop()[1]
        right[i] = count
        stack.append((arr[i], count))
    return sum(a*l*r for a,l,r in zip(arr,left,right)) % mod
```
Classic contribution trick.


67 Sum of Subarray Maximums
Just invert comparison signs.
```
def sum_subarray_maxs(arr):
    n = len(arr)
    mod = 10**9 + 7
    left, right = [0]*n, [0]*n
    stack = []
    for i, x in enumerate(arr):
        count = 1
        while stack and stack[-1][0] < x:
            count += stack.pop()[1]
        left[i] = count
        stack.append((x, count))
    stack.clear()
    for i in range(n-1, -1, -1):
        count = 1
        while stack and stack[-1][0] <= arr[i]:
            count += stack.pop()[1]
        right[i] = count
        stack.append((arr[i], count))
    return sum(a*l*r for a,l,r in zip(arr,left,right)) % mod
```

68 Next Greater Element
Monotonic decreasing stack.
```
def next_greater_elements(arr):
    res = [-1]*len(arr)
    stack = []
    for i, x in enumerate(arr):
        while stack and arr[stack[-1]] < x:
            res[stack.pop()] = x
        stack.append(i)
    return res
```

69 Largest Rectangle in Histogram
Classic monotonic stack O(n).
```
def largest_rectangle_area(heights):
    stack = []
    max_area = 0
    heights.append(0)
    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(i)
    heights.pop()
    return max_area
```

70 Maximal Rectangle (Binary Matrix)
Use histogram per row + above function.
```
def maximal_rectangle(matrix):
    if not matrix: return 0
    n, m = len(matrix), len(matrix[0])
    heights = [0]*m
    best = 0
    for row in matrix:
        for j in range(m):
            heights[j] = heights[j] + 1 if row[j] == 1 else 0
        best = max(best, largest_rectangle_area(heights))
    return best
```
A common combination problem.


71 Rotate Matrix 90° Clockwise (In-Place)
```
def rotate_matrix(mat):
    n = len(mat)
    # transpose
    for i in range(n):
        for j in range(i+1, n):
            mat[i][j], mat[j][i] = mat[j][i], mat[i][j]
    # reverse each row
    for row in mat:
        row.reverse()
    return mat
```
💡 Trick: Transpose + Reverse = 90° rotation.


72 Set Matrix Zeroes (O(1) space)
```
def set_zeroes(mat):
    rows, cols = len(mat), len(mat[0])
    first_row = any(mat[0][j] == 0 for j in range(cols))
    first_col = any(mat[i][0] == 0 for i in range(rows))
    for i in range(1, rows):
        for j in range(1, cols):
            if mat[i][j] == 0:
                mat[i][0] = mat[0][j] = 0
    for i in range(1, rows):
        for j in range(1, cols):
            if mat[i][0] == 0 or mat[0][j] == 0:
                mat[i][j] = 0
    if first_row:
        for j in range(cols): mat[0][j] = 0
    if first_col:
        for i in range(rows): mat[i][0] = 0
    return mat
```
Classic constant-space solution.


73 Spiral Order Traversal
```
def spiral_order(mat):
    res = []
    top, left = 0, 0
    bottom, right = len(mat)-1, len(mat[0])-1
    while top <= bottom and left <= right:
        res += mat[top][left:right+1]; top += 1
        for i in range(top, bottom+1): res.append(mat[i][right])
        right -= 1
        if top <= bottom:
            res += mat[bottom][left:right+1][::-1]; bottom -= 1
        if left <= right:
            for i in range(bottom, top-1, -1): res.append(mat[i][left])
            left += 1
    return res
```

74 Diagonal Traverse (Zig-Zag)
```
def diagonal_traverse(mat):
    n, m = len(mat), len(mat[0])
    res = []
    for s in range(n + m - 1):
        tmp = []
        for i in range(max(0, s - m + 1), min(n, s + 1)):
            tmp.append(mat[i][s - i])
        if s % 2 == 0: tmp.reverse()
        res += tmp
    return res
```

75 Search in Sorted Matrix (Staircase Search)
Matrix sorted row- and column-wise.
```
def search_matrix(mat, target):
    n, m = len(mat), len(mat[0])
    i, j = 0, m - 1
    while i < n and j >= 0:
        if mat[i][j] == target:
            return True
        elif mat[i][j] > target:
            j -= 1
        else:
            i += 1
    return False
```
O(n + m).


76 Maximal Square of 1s
Dynamic programming.
```
def maximal_square(mat):
    n, m = len(mat), len(mat[0])
    dp = [[0]*m for _ in range(n)]
    best = 0
    for i in range(n):
        for j in range(m):
            if mat[i][j] == 1:
                if i == 0 or j == 0:
                    dp[i][j] = 1
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
                best = max(best, dp[i][j])
    return best**2
```

77 Number of Islands (DFS / BFS on 2D Grid)
```
def num_islands(grid):
    if not grid: return 0
    n, m = len(grid), len(grid[0])
    seen = set()
    def dfs(i, j):
        if (i,j) in seen or i<0 or j<0 or i>=n or j>=m or grid[i][j]=='0': return
        seen.add((i,j))
        dfs(i+1,j); dfs(i-1,j); dfs(i,j+1); dfs(i,j-1)
    count = 0
    for i in range(n):
        for j in range(m):
            if grid[i][j]=='1' and (i,j) not in seen:
                dfs(i,j)
                count += 1
    return count
```
Classic flood-fill pattern.


78 Matrix Prefix Sum (2D Fenwick-like)
Precompute to query sub-matrix sum in O(1).
```
def matrix_prefix_sum(mat):
    n, m = len(mat), len(mat[0])
    pref = [[0]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(1, m+1):
            pref[i][j] = mat[i-1][j-1] + pref[i-1][j] + pref[i][j-1] - pref[i-1][j-1]
    def query(x1, y1, x2, y2):
        return pref[x2+1][y2+1] - pref[x1][y2+1] - pref[x2+1][y1] + pref[x1][y1]
    return query
```

79 Rotate Matrix by 180°
```
def rotate_180(mat):
    return [row[::-1] for row in mat[::-1]]
```

80 Minimum Path Sum in Grid
DP (bottom-up).
```
def min_path_sum(grid):
    n, m = len(grid), len(grid[0])
    dp = [[0]*m for _ in range(n)]
    dp[0][0] = grid[0][0]
    for i in range(1, n): dp[i][0] = dp[i-1][0] + grid[i][0]
    for j in range(1, m): dp[0][j] = dp[0][j-1] + grid[0][j]
    for i in range(1, n):
        for j in range(1, m):
            dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])
    return dp[-1][-1]
```

81 Maximum Product Subarray (Dynamic Range)
Track both max and min products (due to negatives).
```
def max_product(nums):
    cur_max = cur_min = ans = nums[0]
    for n in nums[1:]:
        if n < 0:
            cur_max, cur_min = cur_min, cur_max
        cur_max = max(n, cur_max * n)
        cur_min = min(n, cur_min * n)
        ans = max(ans, cur_max)
    return ans
```

82 Count of Subarrays Product Less Than K
Sliding window on product.
```
def num_subarray_product_less_than_k(nums, k):
    if k <= 1: return 0
    prod = 1
    left = count = 0
    for right, val in enumerate(nums):
        prod *= val
        while prod >= k:
            prod //= nums[left]
            left += 1
        count += right - left + 1
    return count
```

83 Median of Two Sorted Arrays
Binary search partition method — O(log(min(n,m))).
```
def find_median_sorted_arrays(A, B):
    if len(A) > len(B): A, B = B, A
    m, n = len(A), len(B)
    half = (m + n + 1) // 2
    lo, hi = 0, m
    while lo <= hi:
        i = (lo + hi) // 2
        j = half - i
        if i < m and B[j-1] > A[i]:
            lo = i + 1
        elif i > 0 and A[i-1] > B[j]:
            hi = i - 1
        else:
            if i == 0: max_left = B[j-1]
            elif j == 0: max_left = A[i-1]
            else: max_left = max(A[i-1], B[j-1])
            if (m + n) % 2: return max_left
            if i == m: min_right = B[j]
            elif j == n: min_right = A[i]
            else: min_right = min(A[i], B[j])
            return (max_left + min_right) / 2
```

84 Longest Increasing Subsequence (Binary Search)
O(n log n) using patience sorting.
```
import bisect
def length_of_LIS(nums):
    tails = []
    for x in nums:
        i = bisect.bisect_left(tails, x)
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)
```

85 Russian Doll Envelopes
Sort and apply LIS on heights.
```
def max_envelopes(envelopes):
    envelopes.sort(key=lambda x: (x[0], -x[1]))
    import bisect
    dp = []
    for _, h in envelopes:
        i = bisect.bisect_left(dp, h)
        if i == len(dp): dp.append(h)
        else: dp[i] = h
    return len(dp)
```

86 Longest Consecutive Sequence (O(n) using HashSet)
```
def longest_consecutive(nums):
    s = set(nums)
    longest = 0
    for n in s:
        if n - 1 not in s:
            cur = n
            streak = 1
            while cur + 1 in s:
                cur += 1
                streak += 1
            longest = max(longest, streak)
    return longest
```

87 Subarray Sum Equals K (Prefix + Hash Map)
Classic prefix-sum pattern.
```
def subarray_sum(nums, k):
    prefix = count = 0
    seen = {0: 1}
    for n in nums:
        prefix += n
        if prefix - k in seen:
            count += seen[prefix - k]
        seen[prefix] = seen.get(prefix, 0) + 1
    return count
```

88 Continuous Subarray Sum (Check Multiple of k)
If prefix sums mod k repeat, subarray sum is multiple of k.
```
def check_subarray_sum(nums, k):
    mods = {0: -1}
    total = 0
    for i, n in enumerate(nums):
        total += n
        if k != 0:
            total %= k
        if total in mods:
            if i - mods[total] > 1:
                return True
        else:
            mods[total] = i
    return False
```

89 Find Peak Element (Binary Search)
```
def find_peak(nums):
    l, r = 0, len(nums) - 1
    while l < r:
        mid = (l + r) // 2
        if nums[mid] < nums[mid + 1]:
            l = mid + 1
        else:
            r = mid
    return l
```

90 Kth Largest Element in an Array (Quickselect)
```
import random
def find_kth_largest(nums, k):
    def quickselect(l, r):
        pivot = nums[random.randint(l, r)]
        left, mid, right = [], [], []
        for n in nums[l:r+1]:
            if n > pivot: left.append(n)
            elif n < pivot: right.append(n)
            else: mid.append(n)
        if k <= len(left):
            nums[l:r+1] = left
            return quickselect(l, l + len(left) - 1)
        elif k > len(left) + len(mid):
            nums[l:r+1] = right
            k2 = k - len(left) - len(mid)
            return quickselect(r - len(right) + 1, r)
        else:
            return pivot
    return quickselect(0, len(nums) - 1)
```

91 Trapping Rain Water II (3D Heap / Matrix)
Water trapped in 2D elevation map.
```
import heapq
def trap_rain_water(heightMap):
    if not heightMap or not heightMap[0]: return 0
    n, m = len(heightMap), len(heightMap[0])
    heap = []
    visited = [[False]*m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            if i==0 or j==0 or i==n-1 or j==m-1:
                heapq.heappush(heap, (heightMap[i][j], i, j))
                visited[i][j] = True
    res = 0
    dirs = [(0,1),(1,0),(-1,0),(0,-1)]
    while heap:
        h, x, y = heapq.heappop(heap)
        for dx, dy in dirs:
            nx, ny = x+dx, y+dy
            if 0<=nx<n and 0<=ny<m and not visited[nx][ny]:
                res += max(0, h - heightMap[nx][ny])
                heapq.heappush(heap, (max(h, heightMap[nx][ny]), nx, ny))
                visited[nx][ny] = True
    return res
```

92 Max Sum Rectangle in 2D Matrix
Kadane on 2D matrix.
```
def max_sum_rectangle(mat):
    n, m = len(mat), len(mat[0])
    best = float('-inf')
    for top in range(n):
        row_sum = [0]*m
        for bottom in range(top, n):
            for j in range(m): row_sum[j] += mat[bottom][j]
            # Kadane on row_sum
            cur = 0
            for val in row_sum:
                cur = max(val, cur + val)
                best = max(best, cur)
    return best
```

93 Submatrix Sum Equals K
Prefix sum + hash map trick.
```
def num_submatrix_sum_target(mat, target):
    n, m = len(mat), len(mat[0])
    for row in mat:
        for j in range(1,m): row[j] += row[j-1]
    count = 0
    for left in range(m):
        for right in range(left, m):
            sums = {0:1}
            cur = 0
            for i in range(n):
                cur += mat[i][right] - (mat[i][left-1] if left > 0 else 0)
                count += sums.get(cur - target, 0)
                sums[cur] = sums.get(cur, 0) + 1
    return count
```

94 Merge Intervals (Advanced)
```
def merge_intervals(intervals):
    intervals.sort(key=lambda x: x[0])
    merged = []
    for s,e in intervals:
        if not merged or merged[-1][1] < s:
            merged.append([s,e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return merged
```

95 Median of Sliding Window
```
import heapq
from bisect import insort, bisect_left
def median_sliding_window(nums, k):
    window = sorted(nums[:k])
    res = []
    for i in range(k, len(nums)+1):
        median = window[k//2] if k%2 else (window[k//2-1]+window[k//2])/2
        res.append(median)
        if i < len(nums):
            # remove nums[i-k], insert nums[i]
            idx = bisect_left(window, nums[i-k])
            window.pop(idx)
            insort(window, nums[i])
    return res
```

96 Largest Number (Custom Sort)
```
from functools import cmp_to_key
def largest_number(nums):
    nums = list(map(str, nums))
    nums.sort(key=cmp_to_key(lambda x,y: 1 if x+y<y+x else -1))
    res = ''.join(nums)
    return res if res[0] != '0' else '0'
```

97 Shortest Unsorted Continuous Subarray
```
def find_unsorted_subarray(nums):
    sorted_nums = sorted(nums)
    l = 0
    while l < len(nums) and nums[l] == sorted_nums[l]: l += 1
    r = len(nums) - 1
    while r >= 0 and nums[r] == sorted_nums[r]: r -= 1
    return 0 if l > r else r - l + 1
```

98 Maximum Circular Subarray Sum
Kadane + total sum trick.
```
def max_circular_subarray_sum(nums):
    def kadane(arr):
        max_end = max_so_far = arr[0]
        for x in arr[1:]:
            max_end = max(x, max_end + x)
            max_so_far = max(max_so_far, max_end)
        return max_so_far
    max_kadane = kadane(nums)
    total = sum(nums)
    min_kadane = kadane([-x for x in nums])
    max_wrap = total + min_kadane
    return max(max_kadane, max_wrap) if max_wrap != 0 else max_kadane
```

99 Count of Smaller Numbers After Self
```
def count_smaller(nums):
    res = []
    sorted_list = []
    from bisect import bisect_left, insort
    for x in reversed(nums):
        idx = bisect_left(sorted_list, x)
        res.append(idx)
        insort(sorted_list, x)
    return res[::-1]
```

100 The Skyline Problem (Heap Sweep)
```
import heapq
def get_skyline(buildings):
    events = [(L,-H,R) for L,R,H in buildings] + [(R,0,0) for L,R,H in buildings]
    events.sort()
    res = []
    heap = [(0,float('inf'))]
    for x, negH, R in events:
        while x >= heap[0][1]: heapq.heappop(heap)
        if negH != 0: heapq.heappush(heap, (negH, R))
        if res[-1][1] if res else 0 != -heap[0][0]:
            res.append([x, -heap[0][0]])
    return res
```
🔥 Hard sweep-line + heap pattern.