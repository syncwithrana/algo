function max_profit(jobs, ind, last) {
    if (ind === jobs.length)
        return 0;

    let ans = max_profit(jobs, ind + 1, last);

    if (jobs[ind][0] >= last)
        ans = Math.max(ans, jobs[ind][2] + 
        max_profit(jobs, ind + 1, jobs[ind][1]));

    return ans;
}

// Driver Code
const jobs = [
    [1, 2, 50],
    [3, 5, 20],
    [6, 19, 100],
    [2, 100, 200]
];
jobs.sort((a, b) => a[0] - b[0]);
console.log(max_profit(jobs, 0, -1));