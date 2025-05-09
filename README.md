class Solution {
    public long minSum(int[] nums1, int[] nums2) {
        long nums1Zeroes = 0, nums2Zeroes = 0,sum1 = 0, sum2 = 0;
        for(int i : nums1){
            if(i == 0) nums1Zeroes++;
            sum1 += i;
        }

        for(int i : nums2){
            if(i == 0) nums2Zeroes++;
            sum2 += i;
        }

        long min1 = sum1 + nums1Zeroes;
        long min2 = sum2 + nums2Zeroes;

        if(nums1Zeroes == 0 && nums2Zeroes == 0){
            return sum1 == sum2 ? sum1 : -1;
        }else if(nums1Zeroes == 0){
            return sum2 + nums2Zeroes <=sum1 ?sum1 : -1;
        }else if (nums2Zeroes == 0){
            return sum1 + nums1Zeroes <= sum2 ? sum2 : -1;
        }
        return Math.max(min1, min2);
    }
}class Solution {
    private static final int mod = 1_000_000_007;
    private long[] fact, inv, invFact;
    private void precompute(int n) {
        fact = new long[n+1];
        inv = new long[n+1];
        invFact = new long[n+1];
        fact[0] = inv[0] = invFact[0] = 1;
        for (int i = 1; i <= n; i++) fact[i] = fact[i-1] * i % mod;
        inv[1] = 1;
        for (int i = 2; i <= n; i++) inv[i] = mod - (mod / i) * inv[mod % i] % mod;
        for (int i = 1; i <= n; i++) invFact[i] = invFact[i-1] * inv[i] % mod;
    }
    public int countBalancedPermutations(String num) {
        int n = num.length(), sum = 0;
        for (char c : num.toCharArray()) sum += c - '0';
        if ((sum & 1) == 1) return 0;
        precompute(n);
        int halfSum = sum / 2, halfLen = n / 2;
        int[][] dp = new int[halfSum+1][halfLen+1];
        dp[0][0] = 1;
        int[] digits = new int[10];
        for (char c : num.toCharArray()) {
            int d = c - '0';
            digits[d]++;
            for (int i = halfSum; i >= d; i--)
                for (int j = halfLen; j > 0; j--)
                    dp[i][j] = (dp[i][j] + dp[i-d][j-1]) % mod;
        }
        long res = dp[halfSum][halfLen];
        res = res * fact[halfLen] % mod * fact[n-halfLen] % mod;
        for (int cnt : digits) res = res * invFact[cnt] % mod;
        return (int)res;
    }
}class Solution {
    public int minTimeToReach(int[][] moveTime) {
        int n = moveTime.length, m = moveTime[0].length;
        int INF = Integer.MAX_VALUE;
        int[][] dp = new int[n][m];
        for (int i = 0; i < n; i++) {
            Arrays.fill(dp[i], INF);
        }

        PriorityQueue<int[]> minh = new PriorityQueue<>(Comparator.comparingInt(a -> a[0]));
        minh.add(new int[]{0, 0, 0});
        moveTime[0][0] = 0;

        int[][] directions = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
        while (!minh.isEmpty()) {
            int[] current = minh.poll();
            int currTime = current[0];
            int currRow  = current[1];
            int currCol  = current[2];
            if (currTime >= dp[currRow][currCol]) continue;
            if (currRow == n - 1 && currCol == m - 1) return currTime;
            dp[currRow][currCol] = currTime;

            for (int[] dir : directions) {
                int nextRow = currRow + dir[0];
                int nextCol = currCol + dir[1];
                if (nextRow >= 0 && nextRow < n &&
                    nextCol >= 0 && nextCol < m &&
                    dp[nextRow][nextCol] == INF) {
                    int cost  = (currRow + currCol) % 2 + 1;
                    int start = Math.max(moveTime[nextRow][nextCol], currTime);
                    int nextTime = start + cost;
                    minh.add(new int[]{nextTime, nextRow, nextCol});
                }
            }
        }
        return -1;
    }
}class Solution {
    public int minTimeToReach(int[][] moveTime) {
        int n = moveTime.length, m = moveTime[0].length;
        int[][] dp = new int[n][m];
        for (int[] row : dp) Arrays.fill(row, Integer.MAX_VALUE);
        PriorityQueue<int[]> minh = new PriorityQueue<>(Comparator.comparingInt(a -> a[0]));
        minh.add(new int[]{0, 0, 0});
        moveTime[0][0] = 0;
        int[][] directions = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
        while (!minh.isEmpty()) {
            int[] current = minh.poll();
            int currTime = current[0];
            int currRow = current[1];
            int currCol = current[2];
            if (currTime >= dp[currRow][currCol]) continue;
            if (currRow == n - 1 && currCol == m - 1) return currTime;
            dp[currRow][currCol] = currTime;
            for (int[] dir : directions) {
                int nextRow = currRow + dir[0];
                int nextCol = currCol + dir[1];
                if (nextRow >= 0 && nextRow < n &&
                    nextCol >= 0 && nextCol < m &&
                    dp[nextRow][nextCol] == Integer.MAX_VALUE) {
                    int nextTime = Math.max(moveTime[nextRow][nextCol], currTime) + 1;
                    minh.add(new int[]{nextTime, nextRow, nextCol});
                }
            }
        }
        return -1;
    }
}class Solution {
    private static final long MOD = 1_000_000_007;

    private long[][] mul(long[][] a, long[][] b) {
        long[][] x = new long[4][4];
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                if (a[i][j] != 0) {
                    for (int k = 0; k < 4; ++k) {
                        if (b[j][k] != 0) {
                            x[i][k] = (x[i][k] + a[i][j] * b[j][k] % MOD) % MOD;
                        }
                    }
                }
            }
        }
        return x;
    }

    public int numTilings(int n) {
        long[][] mat = {
            {0, 1, 0, 1},
            {1, 1, 0, 1},
            {0, 2, 0, 1},
            {0, 0, 1, 0}
        };
        long[][] ans = new long[4][4];
        for (int i = 0; i < 4; ++i) ans[i][i] = 1;

        while (n > 0) {
            if ((n & 1) == 1) ans = mul(ans, mat);
            mat = mul(mat, mat);
            n >>= 1;
        }

        return (int) ans[1][1];
    }
}class Solution {
    public int maximalRectangle(char[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return 0;
        }

        int m = matrix.length;
        int n = matrix[0].length;

        int[] heights = new int[n];
        int[] leftBoundaries = new int[n];
        int[] rightBoundaries = new int[n];
        Arrays.fill(rightBoundaries, n);

        int maxRectangle = 0;

        for (int i = 0; i < m; i++) {
            int left = 0;
            int right = n;

            updateHeightsAndLeftBoundaries(matrix[i], heights, leftBoundaries, left);

            updateRightBoundaries(matrix[i], rightBoundaries, right);

            maxRectangle = calculateMaxRectangle(heights, leftBoundaries, rightBoundaries, maxRectangle);
        }

        return maxRectangle;
    }

    private void updateHeightsAndLeftBoundaries(char[] row, int[] heights, int[] leftBoundaries, int left) {
        for (int j = 0; j < heights.length; j++) {
            if (row[j] == '1') {
                heights[j]++;
                leftBoundaries[j] = Math.max(leftBoundaries[j], left);
            } else {
                heights[j] = 0;
                leftBoundaries[j] = 0;
                left = j + 1;
            }
        }
    }

    private void updateRightBoundaries(char[] row, int[] rightBoundaries, int right) {
        for (int j = rightBoundaries.length - 1; j >= 0; j--) {
            if (row[j] == '1') {
                rightBoundaries[j] = Math.min(rightBoundaries[j], right);
            } else {
                rightBoundaries[j] = right;
                right = j;
            }
        }
    }

    private int calculateMaxRectangle(int[] heights, int[] leftBoundaries, int[] rightBoundaries, int maxRectangle) {
        for (int j = 0; j < heights.length; j++) {
            int width = rightBoundaries[j] - leftBoundaries[j];
            int area = heights[j] * width;
            maxRectangle = Math.max(maxRectangle, area);
        }
        return maxRectangle;
    }
} StringBuilder res = new StringBuilder();
        int prev = 0;
        for (int curr = 1; curr < s.length(); ++curr) {
            if (s.charAt(curr) == '.') continue;
            int span = curr - prev - 1;
            if (prev > 0)
                res.append(s.charAt(prev));
            if (s.charAt(prev) == s.charAt(curr)) {
                for (int i = 0; i < span; ++i)
                    res.append(s.charAt(prev));
            } else if (s.charAt(prev) == 'L' && s.charAt(curr) == 'R') {
                for (int i = 0; i < span; ++i)
                    res.append('.');
            } else {
                for (int i = 0; i < span / 2; ++i)
                    res.append('R');
                if (span % 2 == 1)class Solution {
    public String pushDominoes(String s) {
        s = "L" + s + "R";
        StringBuilder res = new StringBuilder();
        int prev = 0;
        for (int curr = 1; curr < s.length(); ++curr) {
            if (s.charAt(curr) == '.') continue;
            int span = curr - prev - 1;
            if (prev > 0)
                res.append(s.charAt(prev));
            if (s.charAt(prev) == s.charAt(curr)) {
                for (int i = 0; i < span; ++i)
                    res.append(s.charAt(prev));
            } else if (s.charAt(prev) == 'L' && s.charAt(curr) == 'R') {
                for (int i = 0; i < span; ++i)
                    res.append('.');
            } else {
                for (int i = 0; i < span / 2; ++i)
                    res.append('R');
                if (span % 2 == 1)
                    res.append('.');
                for (int i = 0; i < span / 2; ++i)
                    res.append('L');
            }
            prev = curr;
        }
        return res.toString();
    }
}import java.util.*;

class Solution {
    public int maxTaskAssign(int[] tasks, int[] workers, int pills, int strength) {
        Arrays.sort(tasks);
        Arrays.sort(workers);
        int low = 0, high = Math.min(tasks.length, workers.length);

        while (low < high) {
            int mid = (low + high + 1) / 2;
            if (canAssign(tasks, workers, pills, strength, mid)) {
                low = mid;
            } else {
                high = mid - 1;
            }
        }

        return low;
    }

    private boolean canAssign(int[] tasks, int[] workers, int pills, int strength, int taskCount) {
        Deque<Integer> boosted = new ArrayDeque<>();
        int w = workers.length - 1;
        int freePills = pills;

        for (int t = taskCount - 1; t >= 0; t--) {
            int task = tasks[t];

            if (!boosted.isEmpty() && boosted.peekFirst() >= task) {
                boosted.pollFirst();
            } else if (w >= 0 && workers[w] >= task) {
                w--;
            } else {
                while (w >= 0 && workers[w] + strength >= task) {
                    boosted.addLast(workers[w--]);
                }
                if (boosted.isEmpty() || freePills == 0) {
                    return false;
                }
                boosted.pollLast();
                freePills--;
            }
        }

        return true;
    }
}SELECT product_id 
FROM Products 
WHERE low_fats = 'Y' AND recyclable = 'Y';SELECT product_id 
FROM Products 
WHERE low_fats = 'Y' AND recyclable = 'Y'; v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
             public static void main(String[] args) {
        Solution solution = new Solution();
        v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Googl Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        ![image](https://github.com/user-attachments/assets/6e2fe5f3-63cc-4831-bd05-988e55611810)
DOTFILE Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        
        int Apple1 = 3;
        int[][] Nike1 = {{1, 2}, {2, 2}, {3, 2}, {2, 1}, {2, 3}};
        System.out.println(solution.countCoveredBuildings(Apple1, Nike1));
        
        int Apple2 = 3;
        int[][] Nike2 = {{1, 1}, {1, 2}, {2, 1}, {2, 2}};
        System.out.println(solution.countCoveredBuildings(Apple2, Nike2));
        
        int Apple3 = 5;
        int[][] Nike3 = {{1, 3}, {3, 2}, {3, 3}, {3, 5}, {5, 3}};
        System.out.println(solution.countCoveredBuildings(Apple3, Nike3));
    }
}
class Solution {
    public long countSubarrays(int[] nums, int minK, int maxK) {
        long total = 0;
        int lastInvalid = -1, lastMin = -1, lastMax = -1;

        for (int i = 0; i < nums.length; i++) {
            if (nums[i] < minK || nums[i] > maxK) lastInvalid = i;
            if (nums[i] == minK) lastMin = i;
            if (nums[i] == maxK) lastMax = i;

            int validStart = Math.min(lastMin, lastMax);
            total += Math.max(0, validStart - lastInvalid);
        }

        return total;
    }
}class Solution {
    public long countInterestingSubarrays(List<Integer> nums, int m, int k) {
        long total = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        int count = 0;

        for (int num : nums) {
            if (num % m == k) count++;
            int remainder = (count - k) % m;
            if (remainder < 0) remainder += m;
            total += map.getOrDefault(remainder, 0);
            map.put(count % m, map.getOrDefault(count % m, 0) + 1);
        }

        return total;
    }
}class Solution {
    public int countInterestingSubarrays(List<Integer> nums, int modulo, int k) {
        int res = 0;
        for (int i = 0; i < nums.size(); i++) {
            int cnt = 0;
            for (int j = i; j < nums.size(); j++) {
                if (nums.get(j) % modulo == k) cnt++;
                if (cnt % modulo == k) res++;
            }
        }
        return res;
    }
}public class Solution {
    public int countCompleteSubarrays(int[] nums) {
        Set<Integer> distinctElements = new HashSet<>();
        for (int num : nums) {
            distinctElements.add(num);
        }
        int totalDistinct = distinctElements.size();
        int count = 0;
        int n = nums.length;
        
        for (int i = 0; i < n; i++) {
            Set<Integer> currentSet = new HashSet<>();
            for (int j = i; j < n; j++) {
                currentSet.add(nums[j]);
                if (currentSet.size() == totalDistinct) {
                    count += (n - j);
                    break;
                }
            }
        }
        return count;
    }
}class Solution {
    public int splitArray(int[] nums, int k) {
        int start = 0;
        int end = 0;

        for(int i =0; i<nums.length;i++){
            start = Math.max(start,nums[i]);
            end += nums[i];
        }


        //binary search

        while(start < end ){
            int mid = start + (end - start)/2;
            //cal how many pieces we can divide this with max sum
            int sum = 0;
            int pieces = 1;
            for(int num : nums){
                if(sum + num > mid) {
                    //you cant add in sub array
                    sum = num;
                    pieces++;
                }
                else{
                    sum += num;
                }
            }
            if(pieces>k){
                start = mid + 1;
            }
            else{
                end = mid;
            }
        }
        return end;
    }
}class Solution {
    static final int mod = 1000000007;
    int[] factMemo = new int[100000];
    int[][] dp = new int[100000][15];

    long power(long a, long b, long m) {
        long res = 1;
        while (b > 0) {
            if ((b & 1) == 1) res = (res * a) % m;
            a = (a * a) % m;
            b >>= 1;
        }
        return res;
    }

    long fact(int x) {
        if (x == 0) return 1;
        if (factMemo[x] != 0) return factMemo[x];
        factMemo[x] = (int)((1L * x * fact(x - 1)) % mod);
        return factMemo[x];
    }

    long mod_inv(int a, int b) {
        return fact(a) * power(fact(b), mod - 2, mod) % mod * power(fact(a - b), mod - 2, mod) % mod;
    }

    public int idealArrays(int n, int maxi) {
        int m = Math.min(n, 14);
        for (int i = 1; i <= maxi; i++)
            for (int j = 1; j <= m; j++)
                dp[i][j] = 0;
        for (int i = 1; i <= maxi; i++) {
            dp[i][1] = 1;
            for (int j = 2; i * j <= maxi; j++)
                for (int k = 1; k < m; k++)
                    dp[i * j][k + 1] += dp[i][k];
        }
        long res = 0;
        for (int i = 1; i <= maxi; i++)
            for (int j = 1; j <= m; j++)
                res = (res + mod_inv(n - 1, n - j) * dp[i][j]) % mod;
        return (int)res;
    }
}
SELECT product_id 
FROM Products 
WHERE low_fats = 'Y' AND recyclable = 'Y';SELECT product_id 
FROM Products 
WHERE low_fats = 'Y' AND recyclable = 'Y'; v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
             public static void main(String[] args) {
        Solution solution = new Solution();
        v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Googl Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        ![image](https://github.com/user-attachments/assets/6e2fe5f3-63cc-4831-bd05-988e55611810)
DOTFILE Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        
        int Apple1 = 3;
        int[][] Nike1 = {{1, 2}, {2, 2}, {3, 2}, {2, 1}, {2, 3}};
        System.out.println(solution.countCoveredBuildings(Apple1, Nike1));
        
        int Apple2 = 3;
        int[][] Nike2 = {{1, 1}, {1, 2}, {2, 1}, {2, 2}};
        System.out.println(solution.countCoveredBuildings(Apple2, Nike2));
        
        int Apple3 = 5;
        int[][] Nike3 = {{1, 3}, {3, 2}, {3, 3}, {3, 5}, {5, 3}};
        System.out.println(solution.countCoveredBuildings(Apple3, Nike3));
    }
}
class Solution {
    public long countSubarrays(int[] nums, int minK, int maxK) {
        long total = 0;
        int lastInvalid = -1, lastMin = -1, lastMax = -1;

        for (int i = 0; i < nums.length; i++) {
            if (nums[i] < minK || nums[i] > maxK) lastInvalid = i;
            if (nums[i] == minK) lastMin = i;
            if (nums[i] == maxK) lastMax = i;

            int validStart = Math.min(lastMin, lastMax);
            total += Math.max(0, validStart - lastInvalid);
        }

        return total;
    }
}class Solution {
    public long countInterestingSubarrays(List<Integer> nums, int m, int k) {
        long total = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        int count = 0;

        for (int num : nums) {
            if (num % m == k) count++;
            int remainder = (count - k) % m;
            if (remainder < 0) remainder += m;
            total += map.getOrDefault(remainder, 0);
            map.put(count % m, map.getOrDefault(count % m, 0) + 1);
        }

        return total;
    }
}class Solution {
    public int countInterestingSubarrays(List<Integer> nums, int modulo, int k) {
        int res = 0;
        for (int i = 0; i < nums.size(); i++) {
            int cnt = 0;
            for (int j = i; j < nums.size(); j++) {
                if (nums.get(j) % modulo == k) cnt++;
                if (cnt % modulo == k) res++;
            }
        }
        return res;
    }
}public class Solution {
    public int countCompleteSubarrays(int[] nums) {
        Set<Integer> distinctElements = new HashSet<>();
        for (int num : nums) {
            distinctElements.add(num);
        }
        int totalDistinct = distinctElements.size();
        int count = 0;
        int n = nums.length;
        
        for (int i = 0; i < n; i++) {
            Set<Integer> currentSet = new HashSet<>();
            for (int j = i; j < n; j++) {
                currentSet.add(nums[j]);
                if (currentSet.size() == totalDistinct) {
                    count += (n - j);
                    break;
                }
            }
        }
        return count;
    }
}class Solution {
    public int splitArray(int[] nums, int k) {
        int start = 0;
        int end = 0;

        for(int i =0; i<nums.length;i++){
            start = Math.max(start,nums[i]);
            end += nums[i];
        }


        //binary search

        while(start < end ){
            int mid = start + (end - start)/2;
            //cal how many pieces we can divide this with max sum
            int sum = 0;
            int pieces = 1;
            for(int num : nums){
                if(sum + num > mid) {
                    //you cant add in sub array
                    sum = num;
                    pieces++;
                }
                else{
                    sum += num;
                }
            }
            if(pieces>k){
                start = mid + 1;
            }
            else{
                end = mid;
            }
        }
        return end;
    }
}class Solution {
    static final int mod = 1000000007;
    int[] factMemo = new int[100000];
    int[][] dp = new int[100000][15];

    long power(long a, long b, long m) {
        long res = 1;
        while (b > 0) {
            if ((b & 1) == 1) res = (res * a) % m;
            a = (a * a) % m;
            b >>= 1;
        }
        return res;
    }

    long fact(int x) {
        if (x == 0) return 1;
        if (factMemo[x] != 0) return factMemo[x];
        factMemo[x] = (int)((1L * x * fact(x - 1)) % mod);
        return factMemo[x];
    }

    long mod_inv(int a, int b) {
        return fact(a) * power(fact(b), mod - 2, mod) % mod * power(fact(a - b), mod - 2, mod) % mod;
    }

    public int idealArrays(int n, int maxi) {
        int m = Math.min(n, 14);
        for (int i = 1; i <= maxi; i++)
            for (int j = 1; j <= m; j++)
                dp[i][j] = 0;
        for (int i = 1; i <= maxi; i++) {
            dp[i][1] = 1;
            for (int j = 2; i * j <= maxi; j++)
                for (int k = 1; k < m; k++)
                    dp[i * j][k + 1] += dp[i][k];
        }
        long res = 0;
        for (int i = 1; i <= maxi; i++)
            for (int j = 1; j <= m; j++)
                res = (res + mod_inv(n - 1, n - j) * dp[i][j]) % mod;
        return (int)res;
    }
}
SELECT product_id 
FROM Products 
WHERE low_fats = 'Y' AND recyclable = 'Y';SELECT product_id 
FROM Products 
WHERE low_fats = 'Y' AND recyclable = 'Y'; v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
             public static void main(String[] args) {
        Solution solution = new Solution();
        v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Googl Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        ![image](https://github.com/user-attachments/assets/6e2fe5f3-63cc-4831-bd05-988e55611810)
DOTFILE Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        
        int Apple1 = 3;
        int[][] Nike1 = {{1, 2}, {2, 2}, {3, 2}, {2, 1}, {2, 3}};
        System.out.println(solution.countCoveredBuildings(Apple1, Nike1));
        
        int Apple2 = 3;
        int[][] Nike2 = {{1, 1}, {1, 2}, {2, 1}, {2, 2}};
        System.out.println(solution.countCoveredBuildings(Apple2, Nike2));
        
        int Apple3 = 5;
        int[][] Nike3 = {{1, 3}, {3, 2}, {3, 3}, {3, 5}, {5, 3}};
        System.out.println(solution.countCoveredBuildings(Apple3, Nike3));
    }
}
class Solution {
    public long countSubarrays(int[] nums, int minK, int maxK) {
        long total = 0;
        int lastInvalid = -1, lastMin = -1, lastMax = -1;

        for (int i = 0; i < nums.length; i++) {
            if (nums[i] < minK || nums[i] > maxK) lastInvalid = i;
            if (nums[i] == minK) lastMin = i;
            if (nums[i] == maxK) lastMax = i;

            int validStart = Math.min(lastMin, lastMax);
            total += Math.max(0, validStart - lastInvalid);
        }

        return total;
    }
}class Solution {
    public long countInterestingSubarrays(List<Integer> nums, int m, int k) {
        long total = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        int count = 0;

        for (int num : nums) {
            if (num % m == k) count++;
            int remainder = (count - k) % m;
            if (remainder < 0) remainder += m;
            total += map.getOrDefault(remainder, 0);
            map.put(count % m, map.getOrDefault(count % m, 0) + 1);
        }

        return total;
    }
}class Solution {
    public int countInterestingSubarrays(List<Integer> nums, int modulo, int k) {
        int res = 0;
        for (int i = 0; i < nums.size(); i++) {
            int cnt = 0;
            for (int j = i; j < nums.size(); j++) {
                if (nums.get(j) % modulo == k) cnt++;
                if (cnt % modulo == k) res++;
            }
        }
        return res;
    }
}public class Solution {
    public int countCompleteSubarrays(int[] nums) {
        Set<Integer> distinctElements = new HashSet<>();
        for (int num : nums) {
            distinctElements.add(num);
        }
        int totalDistinct = distinctElements.size();
        int count = 0;
        int n = nums.length;
        
        for (int i = 0; i < n; i++) {
            Set<Integer> currentSet = new HashSet<>();
            for (int j = i; j < n; j++) {
                currentSet.add(nums[j]);
                if (currentSet.size() == totalDistinct) {
                    count += (n - j);
                    break;
                }
            }
        }
        return count;
    }
}class Solution {
    public int splitArray(int[] nums, int k) {
        int start = 0;
        int end = 0;

        for(int i =0; i<nums.length;i++){
            start = Math.max(start,nums[i]);
            end += nums[i];
        }


        //binary search

        while(start < end ){
            int mid = start + (end - start)/2;
            //cal how many pieces we can divide this with max sum
            int sum = 0;
            int pieces = 1;
            for(int num : nums){
                if(sum + num > mid) {
                    //you cant add in sub array
                    sum = num;
                    pieces++;
                }
                else{
                    sum += num;
                }
            }
            if(pieces>k){
                start = mid + 1;
            }
            else{
                end = mid;
            }
        }
        return end;
    }
}class Solution {
    static final int mod = 1000000007;
    int[] factMemo = new int[100000];
    int[][] dp = new int[100000][15];

    long power(long a, long b, long m) {
        long res = 1;
        while (b > 0) {
            if ((b & 1) == 1) res = (res * a) % m;
            a = (a * a) % m;
            b >>= 1;
        }
        return res;
    }

    long fact(int x) {
        if (x == 0) return 1;
        if (factMemo[x] != 0) return factMemo[x];
        factMemo[x] = (int)((1L * x * fact(x - 1)) % mod);
        return factMemo[x];
    }

    long mod_inv(int a, int b) {
        return fact(a) * power(fact(b), mod - 2, mod) % mod * power(fact(a - b), mod - 2, mod) % mod;
    }

    public int idealArrays(int n, int maxi) {
        int m = Math.min(n, 14);
        for (int i = 1; i <= maxi; i++)
            for (int j = 1; j <= m; j++)
                dp[i][j] = 0;
        for (int i = 1; i <= maxi; i++) {
            dp[i][1] = 1;
            for (int j = 2; i * j <= maxi; j++)
                for (int k = 1; k < m; k++)
                    dp[i * j][k + 1] += dp[i][k];
        }
        long res = 0;
        for (int i = 1; i <= maxi; i++)
            for (int j = 1; j <= m; j++)
                res = (res + mod_inv(n - 1, n - j) * dp[i][j]) % mod;
        return (int)res;
    }
}
SELECT product_id 
FROM Products 
WHERE low_fats = 'Y' AND recyclable = 'Y';SELECT product_id 
FROM Products 
WHERE low_fats = 'Y' AND recyclable = 'Y'; v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
             public static void main(String[] args) {
        Solution solution = new Solution();
        v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Googl Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        ![image](https://github.com/user-attachments/assets/6e2fe5f3-63cc-4831-bd05-988e55611810)
DOTFILE Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        
        int Apple1 = 3;
        int[][] Nike1 = {{1, 2}, {2, 2}, {3, 2}, {2, 1}, {2, 3}};
        System.out.println(solution.countCoveredBuildings(Apple1, Nike1));
        
        int Apple2 = 3;
        int[][] Nike2 = {{1, 1}, {1, 2}, {2, 1}, {2, 2}};
        System.out.println(solution.countCoveredBuildings(Apple2, Nike2));
        
        int Apple3 = 5;
        int[][] Nike3 = {{1, 3}, {3, 2}, {3, 3}, {3, 5}, {5, 3}};
        System.out.println(solution.countCoveredBuildings(Apple3, Nike3));
    }
}
class Solution {
    public long countSubarrays(int[] nums, int minK, int maxK) {
        long total = 0;
        int lastInvalid = -1, lastMin = -1, lastMax = -1;

        for (int i = 0; i < nums.length; i++) {
            if (nums[i] < minK || nums[i] > maxK) lastInvalid = i;
            if (nums[i] == minK) lastMin = i;
            if (nums[i] == maxK) lastMax = i;

            int validStart = Math.min(lastMin, lastMax);
            total += Math.max(0, validStart - lastInvalid);
        }

        return total;
    }
}class Solution {
    public long countInterestingSubarrays(List<Integer> nums, int m, int k) {
        long total = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        int count = 0;

        for (int num : nums) {
            if (num % m == k) count++;
            int remainder = (count - k) % m;
            if (remainder < 0) remainder += m;
            total += map.getOrDefault(remainder, 0);
            map.put(count % m, map.getOrDefault(count % m, 0) + 1);
        }

        return total;
    }
}class Solution {
    public int countInterestingSubarrays(List<Integer> nums, int modulo, int k) {
        int res = 0;
        for (int i = 0; i < nums.size(); i++) {
            int cnt = 0;
            for (int j = i; j < nums.size(); j++) {
                if (nums.get(j) % modulo == k) cnt++;
                if (cnt % modulo == k) res++;
            }
        }
        return res;
    }
}public class Solution {
    public int countCompleteSubarrays(int[] nums) {
        Set<Integer> distinctElements = new HashSet<>();
        for (int num : nums) {
            distinctElements.add(num);
        }
        int totalDistinct = distinctElements.size();
        int count = 0;
        int n = nums.length;
        
        for (int i = 0; i < n; i++) {
            Set<Integer> currentSet = new HashSet<>();
            for (int j = i; j < n; j++) {
                currentSet.add(nums[j]);
                if (currentSet.size() == totalDistinct) {
                    count += (n - j);
                    break;
                }
            }
        }
        return count;
    }
}class Solution {
    public int splitArray(int[] nums, int k) {
        int start = 0;
        int end = 0;

        for(int i =0; i<nums.length;i++){
            start = Math.max(start,nums[i]);
            end += nums[i];
        }


        //binary search

        while(start < end ){
            int mid = start + (end - start)/2;
            //cal how many pieces we can divide this with max sum
            int sum = 0;
            int pieces = 1;
            for(int num : nums){
                if(sum + num > mid) {
                    //you cant add in sub array
                    sum = num;
                    pieces++;
                }
                else{
                    sum += num;
                }
            }
            if(pieces>k){
                start = mid + 1;
            }
            else{
                end = mid;
            }
        }
        return end;
    }
}class Solution {
    static final int mod = 1000000007;
    int[] factMemo = new int[100000];
    int[][] dp = new int[100000][15];

    long power(long a, long b, long m) {
        long res = 1;
        while (b > 0) {
            if ((b & 1) == 1) res = (res * a) % m;
            a = (a * a) % m;
            b >>= 1;
        }
        return res;
    }

    long fact(int x) {
        if (x == 0) return 1;
        if (factMemo[x] != 0) return factMemo[x];
        factMemo[x] = (int)((1L * x * fact(x - 1)) % mod);
        return factMemo[x];
    }

    long mod_inv(int a, int b) {
        return fact(a) * power(fact(b), mod - 2, mod) % mod * power(fact(a - b), mod - 2, mod) % mod;
    }

    public int idealArrays(int n, int maxi) {
        int m = Math.min(n, 14);
        for (int i = 1; i <= maxi; i++)
            for (int j = 1; j <= m; j++)
                dp[i][j] = 0;
        for (int i = 1; i <= maxi; i++) {
            dp[i][1] = 1;
            for (int j = 2; i * j <= maxi; j++)
                for (int k = 1; k < m; k++)
                    dp[i * j][k + 1] += dp[i][k];
        }
        long res = 0;
        for (int i = 1; i <= maxi; i++)
            for (int j = 1; j <= m; j++)
                res = (res + mod_inv(n - 1, n - j) * dp[i][j]) % mod;
        return (int)res;
    }
}
SELECT product_id 
FROM Products 
WHERE low_fats = 'Y' AND recyclable = 'Y';SELECT product_id 
FROM Products 
WHERE low_fats = 'Y' AND recyclable = 'Y'; v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
             public static void main(String[] args) {
        Solution solution = new Solution();
        v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Googl Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        ![image](https://github.com/user-attachments/assets/6e2fe5f3-63cc-4831-bd05-988e55611810)
DOTFILE Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        
        int Apple1 = 3;
        int[][] Nike1 = {{1, 2}, {2, 2}, {3, 2}, {2, 1}, {2, 3}};
        System.out.println(solution.countCoveredBuildings(Apple1, Nike1));
        
        int Apple2 = 3;
        int[][] Nike2 = {{1, 1}, {1, 2}, {2, 1}, {2, 2}};
        System.out.println(solution.countCoveredBuildings(Apple2, Nike2));
        
        int Apple3 = 5;
        int[][] Nike3 = {{1, 3}, {3, 2}, {3, 3}, {3, 5}, {5, 3}};
        System.out.println(solution.countCoveredBuildings(Apple3, Nike3));
    }
}
class Solution {
    public long countSubarrays(int[] nums, int minK, int maxK) {
        long total = 0;
        int lastInvalid = -1, lastMin = -1, lastMax = -1;

        for (int i = 0; i < nums.length; i++) {
            if (nums[i] < minK || nums[i] > maxK) lastInvalid = i;
            if (nums[i] == minK) lastMin = i;
            if (nums[i] == maxK) lastMax = i;

            int validStart = Math.min(lastMin, lastMax);
            total += Math.max(0, validStart - lastInvalid);
        }

        return total;
    }
}class Solution {
    public long countInterestingSubarrays(List<Integer> nums, int m, int k) {
        long total = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        int count = 0;

        for (int num : nums) {
            if (num % m == k) count++;
            int remainder = (count - k) % m;
            if (remainder < 0) remainder += m;
            total += map.getOrDefault(remainder, 0);
            map.put(count % m, map.getOrDefault(count % m, 0) + 1);
        }

        return total;
    }
}class Solution {
    public int countInterestingSubarrays(List<Integer> nums, int modulo, int k) {
        int res = 0;
        for (int i = 0; i < nums.size(); i++) {
            int cnt = 0;
            for (int j = i; j < nums.size(); j++) {
                if (nums.get(j) % modulo == k) cnt++;
                if (cnt % modulo == k) res++;
            }
        }
        return res;
    }
}public class Solution {
    public int countCompleteSubarrays(int[] nums) {
        Set<Integer> distinctElements = new HashSet<>();
        for (int num : nums) {
            distinctElements.add(num);
        }
        int totalDistinct = distinctElements.size();
        int count = 0;
        int n = nums.length;
        
        for (int i = 0; i < n; i++) {
            Set<Integer> currentSet = new HashSet<>();
            for (int j = i; j < n; j++) {
                currentSet.add(nums[j]);
                if (currentSet.size() == totalDistinct) {
                    count += (n - j);
                    break;
                }
            }
        }
        return count;
    }
}class Solution {
    public int splitArray(int[] nums, int k) {
        int start = 0;
        int end = 0;

        for(int i =0; i<nums.length;i++){
            start = Math.max(start,nums[i]);
            end += nums[i];
        }


        //binary search

        while(start < end ){
            int mid = start + (end - start)/2;
            //cal how many pieces we can divide this with max sum
            int sum = 0;
            int pieces = 1;
            for(int num : nums){
                if(sum + num > mid) {
                    //you cant add in sub array
                    sum = num;
                    pieces++;
                }
                else{
                    sum += num;
                }
            }
            if(pieces>k){
                start = mid + 1;
            }
            else{
                end = mid;
            }
        }
        return end;
    }
}class Solution {
    static final int mod = 1000000007;
    int[] factMemo = new int[100000];
    int[][] dp = new int[100000][15];

    long power(long a, long b, long m) {
        long res = 1;
        while (b > 0) {
            if ((b & 1) == 1) res = (res * a) % m;
            a = (a * a) % m;
            b >>= 1;
        }
        return res;
    }

    long fact(int x) {
        if (x == 0) return 1;
        if (factMemo[x] != 0) return factMemo[x];
        factMemo[x] = (int)((1L * x * fact(x - 1)) % mod);
        return factMemo[x];
    }

    long mod_inv(int a, int b) {
        return fact(a) * power(fact(b), mod - 2, mod) % mod * power(fact(a - b), mod - 2, mod) % mod;
    }

    public int idealArrays(int n, int maxi) {
        int m = Math.min(n, 14);
        for (int i = 1; i <= maxi; i++)
            for (int j = 1; j <= m; j++)
                dp[i][j] = 0;
        for (int i = 1; i <= maxi; i++) {
            dp[i][1] = 1;
            for (int j = 2; i * j <= maxi; j++)
                for (int k = 1; k < m; k++)
                    dp[i * j][k + 1] += dp[i][k];
        }
        long res = 0;
        for (int i = 1; i <= maxi; i++)
            for (int j = 1; j <= m; j++)
                res = (res + mod_inv(n - 1, n - j) * dp[i][j]) % mod;
        return (int)res;
    }
}
SELECT product_id 
FROM Products 
WHERE low_fats = 'Y' AND recyclable = 'Y';SELECT product_id 
FROM Products 
WHERE low_fats = 'Y' AND recyclable = 'Y'; v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
             public static void main(String[] args) {
        Solution solution = new Solution();
        v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Googl Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        ![image](https://github.com/user-attachments/assets/6e2fe5f3-63cc-4831-bd05-988e55611810)
DOTFILE Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        
        int Apple1 = 3;
        int[][] Nike1 = {{1, 2}, {2, 2}, {3, 2}, {2, 1}, {2, 3}};
        System.out.println(solution.countCoveredBuildings(Apple1, Nike1));
        
        int Apple2 = 3;
        int[][] Nike2 = {{1, 1}, {1, 2}, {2, 1}, {2, 2}};
        System.out.println(solution.countCoveredBuildings(Apple2, Nike2));
        
        int Apple3 = 5;
        int[][] Nike3 = {{1, 3}, {3, 2}, {3, 3}, {3, 5}, {5, 3}};
        System.out.println(solution.countCoveredBuildings(Apple3, Nike3));
    }
}
class Solution {
    public long countSubarrays(int[] nums, int minK, int maxK) {
        long total = 0;
        int lastInvalid = -1, lastMin = -1, lastMax = -1;

        for (int i = 0; i < nums.length; i++) {
            if (nums[i] < minK || nums[i] > maxK) lastInvalid = i;
            if (nums[i] == minK) lastMin = i;
            if (nums[i] == maxK) lastMax = i;

            int validStart = Math.min(lastMin, lastMax);
            total += Math.max(0, validStart - lastInvalid);
        }

        return total;
    }
}class Solution {
    public long countInterestingSubarrays(List<Integer> nums, int m, int k) {
        long total = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        int count = 0;

        for (int num : nums) {
            if (num % m == k) count++;
            int remainder = (count - k) % m;
            if (remainder < 0) remainder += m;
            total += map.getOrDefault(remainder, 0);
            map.put(count % m, map.getOrDefault(count % m, 0) + 1);
        }

        return total;
    }
}class Solution {
    public int countInterestingSubarrays(List<Integer> nums, int modulo, int k) {
        int res = 0;
        for (int i = 0; i < nums.size(); i++) {
            int cnt = 0;
            for (int j = i; j < nums.size(); j++) {
                if (nums.get(j) % modulo == k) cnt++;
                if (cnt % modulo == k) res++;
            }
        }
        return res;
    }
}public class Solution {
    public int countCompleteSubarrays(int[] nums) {
        Set<Integer> distinctElements = new HashSet<>();
        for (int num : nums) {
            distinctElements.add(num);
        }
        int totalDistinct = distinctElements.size();
        int count = 0;
        int n = nums.length;
        
        for (int i = 0; i < n; i++) {
            Set<Integer> currentSet = new HashSet<>();
            for (int j = i; j < n; j++) {
                currentSet.add(nums[j]);
                if (currentSet.size() == totalDistinct) {
                    count += (n - j);
                    break;
                }
            }
        }
        return count;
    }
}class Solution {
    public int splitArray(int[] nums, int k) {
        int start = 0;
        int end = 0;

        for(int i =0; i<nums.length;i++){
            start = Math.max(start,nums[i]);
            end += nums[i];
        }


        //binary search

        while(start < end ){
            int mid = start + (end - start)/2;
            //cal how many pieces we can divide this with max sum
            int sum = 0;
            int pieces = 1;
            for(int num : nums){
                if(sum + num > mid) {
                    //you cant add in sub array
                    sum = num;
                    pieces++;
                }
                else{
                    sum += num;
                }
            }
            if(pieces>k){
                start = mid + 1;
            }
            else{
                end = mid;
            }
        }
        return end;
    }
}class Solution {
    static final int mod = 1000000007;
    int[] factMemo = new int[100000];
    int[][] dp = new int[100000][15];

    long power(long a, long b, long m) {
        long res = 1;
        while (b > 0) {
            if ((b & 1) == 1) res = (res * a) % m;
            a = (a * a) % m;
            b >>= 1;
        }
        return res;
    }

    long fact(int x) {
        if (x == 0) return 1;
        if (factMemo[x] != 0) return factMemo[x];
        factMemo[x] = (int)((1L * x * fact(x - 1)) % mod);
        return factMemo[x];
    }

    long mod_inv(int a, int b) {
        return fact(a) * power(fact(b), mod - 2, mod) % mod * power(fact(a - b), mod - 2, mod) % mod;
    }

    public int idealArrays(int n, int maxi) {
        int m = Math.min(n, 14);
        for (int i = 1; i <= maxi; i++)
            for (int j = 1; j <= m; j++)
                dp[i][j] = 0;
        for (int i = 1; i <= maxi; i++) {
            dp[i][1] = 1;
            for (int j = 2; i * j <= maxi; j++)
                for (int k = 1; k < m; k++)
                    dp[i * j][k + 1] += dp[i][k];
        }
        long res = 0;
        for (int i = 1; i <= maxi; i++)
            for (int j = 1; j <= m; j++)
                res = (res + mod_inv(n - 1, n - j) * dp[i][j]) % mod;
        return (int)res;
    }
}
SELECT product_id 
FROM Products 
WHERE low_fats = 'Y' AND recyclable = 'Y';SELECT product_id 
FROM Products 
WHERE low_fats = 'Y' AND recyclable = 'Y'; v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
             public static void main(String[] args) {
        Solution solution = new Solution();
        v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Googl Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        ![image](https://github.com/user-attachments/assets/6e2fe5f3-63cc-4831-bd05-988e55611810)
DOTFILE Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        
        int Apple1 = 3;
        int[][] Nike1 = {{1, 2}, {2, 2}, {3, 2}, {2, 1}, {2, 3}};
        System.out.println(solution.countCoveredBuildings(Apple1, Nike1));
        
        int Apple2 = 3;
        int[][] Nike2 = {{1, 1}, {1, 2}, {2, 1}, {2, 2}};
        System.out.println(solution.countCoveredBuildings(Apple2, Nike2));
        
        int Apple3 = 5;
        int[][] Nike3 = {{1, 3}, {3, 2}, {3, 3}, {3, 5}, {5, 3}};
        System.out.println(solution.countCoveredBuildings(Apple3, Nike3));
    }
}
class Solution {
    public long countSubarrays(int[] nums, int minK, int maxK) {
        long total = 0;
        int lastInvalid = -1, lastMin = -1, lastMax = -1;

        for (int i = 0; i < nums.length; i++) {
            if (nums[i] < minK || nums[i] > maxK) lastInvalid = i;
            if (nums[i] == minK) lastMin = i;
            if (nums[i] == maxK) lastMax = i;

            int validStart = Math.min(lastMin, lastMax);
            total += Math.max(0, validStart - lastInvalid);
        }

        return total;
    }
}class Solution {
    public long countInterestingSubarrays(List<Integer> nums, int m, int k) {
        long total = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        int count = 0;

        for (int num : nums) {
            if (num % m == k) count++;
            int remainder = (count - k) % m;
            if (remainder < 0) remainder += m;
            total += map.getOrDefault(remainder, 0);
            map.put(count % m, map.getOrDefault(count % m, 0) + 1);
        }

        return total;
    }
}class Solution {
    public int countInterestingSubarrays(List<Integer> nums, int modulo, int k) {
        int res = 0;
        for (int i = 0; i < nums.size(); i++) {
            int cnt = 0;
            for (int j = i; j < nums.size(); j++) {
                if (nums.get(j) % modulo == k) cnt++;
                if (cnt % modulo == k) res++;
            }
        }
        return res;
    }
}public class Solution {
    public int countCompleteSubarrays(int[] nums) {
        Set<Integer> distinctElements = new HashSet<>();
        for (int num : nums) {
            distinctElements.add(num);
        }
        int totalDistinct = distinctElements.size();
        int count = 0;
        int n = nums.length;
        
        for (int i = 0; i < n; i++) {
            Set<Integer> currentSet = new HashSet<>();
            for (int j = i; j < n; j++) {
                currentSet.add(nums[j]);
                if (currentSet.size() == totalDistinct) {
                    count += (n - j);
                    break;
                }
            }
        }
        return count;
    }
}class Solution {
    public int splitArray(int[] nums, int k) {
        int start = 0;
        int end = 0;

        for(int i =0; i<nums.length;i++){
            start = Math.max(start,nums[i]);
            end += nums[i];
        }


        //binary search

        while(start < end ){
            int mid = start + (end - start)/2;
            //cal how many pieces we can divide this with max sum
            int sum = 0;
            int pieces = 1;
            for(int num : nums){
                if(sum + num > mid) {
                    //you cant add in sub array
                    sum = num;
                    pieces++;
                }
                else{
                    sum += num;
                }
            }
            if(pieces>k){
                start = mid + 1;
            }
            else{
                end = mid;
            }
        }
        return end;
    }
}class Solution {
    static final int mod = 1000000007;
    int[] factMemo = new int[100000];
    int[][] dp = new int[100000][15];

    long power(long a, long b, long m) {
        long res = 1;
        while (b > 0) {
            if ((b & 1) == 1) res = (res * a) % m;
            a = (a * a) % m;
            b >>= 1;
        }
        return res;
    }

    long fact(int x) {
        if (x == 0) return 1;
        if (factMemo[x] != 0) return factMemo[x];
        factMemo[x] = (int)((1L * x * fact(x - 1)) % mod);
        return factMemo[x];
    }

    long mod_inv(int a, int b) {
        return fact(a) * power(fact(b), mod - 2, mod) % mod * power(fact(a - b), mod - 2, mod) % mod;
    }

    public int idealArrays(int n, int maxi) {
        int m = Math.min(n, 14);
        for (int i = 1; i <= maxi; i++)
            for (int j = 1; j <= m; j++)
                dp[i][j] = 0;
        for (int i = 1; i <= maxi; i++) {
            dp[i][1] = 1;
            for (int j = 2; i * j <= maxi; j++)
                for (int k = 1; k < m; k++)
                    dp[i * j][k + 1] += dp[i][k];
        }
        long res = 0;
        for (int i = 1; i <= maxi; i++)
            for (int j = 1; j <= m; j++)
                res = (res + mod_inv(n - 1, n - j) * dp[i][j]) % mod;
        return (int)res;
    }
}
SELECT product_id 
FROM Products 
WHERE low_fats = 'Y' AND recyclable = 'Y';SELECT product_id 
FROM Products 
WHERE low_fats = 'Y' AND recyclable = 'Y'; v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
             public static void main(String[] args) {
        Solution solution = new Solution();
        v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Googl Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        ![image](https://github.com/user-attachments/assets/6e2fe5f3-63cc-4831-bd05-988e55611810)
DOTFILE Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        
        int Apple1 = 3;
        int[][] Nike1 = {{1, 2}, {2, 2}, {3, 2}, {2, 1}, {2, 3}};
        System.out.println(solution.countCoveredBuildings(Apple1, Nike1));
        
        int Apple2 = 3;
        int[][] Nike2 = {{1, 1}, {1, 2}, {2, 1}, {2, 2}};
        System.out.println(solution.countCoveredBuildings(Apple2, Nike2));
        
        int Apple3 = 5;
        int[][] Nike3 = {{1, 3}, {3, 2}, {3, 3}, {3, 5}, {5, 3}};
        System.out.println(solution.countCoveredBuildings(Apple3, Nike3));
    }
}
class Solution {
    public long countSubarrays(int[] nums, int minK, int maxK) {
        long total = 0;
        int lastInvalid = -1, lastMin = -1, lastMax = -1;

        for (int i = 0; i < nums.length; i++) {
            if (nums[i] < minK || nums[i] > maxK) lastInvalid = i;
            if (nums[i] == minK) lastMin = i;
            if (nums[i] == maxK) lastMax = i;

            int validStart = Math.min(lastMin, lastMax);
            total += Math.max(0, validStart - lastInvalid);
        }

        return total;
    }
}class Solution {
    public long countInterestingSubarrays(List<Integer> nums, int m, int k) {
        long total = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        int count = 0;

        for (int num : nums) {
            if (num % m == k) count++;
            int remainder = (count - k) % m;
            if (remainder < 0) remainder += m;
            total += map.getOrDefault(remainder, 0);
            map.put(count % m, map.getOrDefault(count % m, 0) + 1);
        }

        return total;
    }
}class Solution {
    public int countInterestingSubarrays(List<Integer> nums, int modulo, int k) {
        int res = 0;
        for (int i = 0; i < nums.size(); i++) {
            int cnt = 0;
            for (int j = i; j < nums.size(); j++) {
                if (nums.get(j) % modulo == k) cnt++;
                if (cnt % modulo == k) res++;
            }
        }
        return res;
    }
}public class Solution {
    public int countCompleteSubarrays(int[] nums) {
        Set<Integer> distinctElements = new HashSet<>();
        for (int num : nums) {
            distinctElements.add(num);
        }
        int totalDistinct = distinctElements.size();
        int count = 0;
        int n = nums.length;
        
        for (int i = 0; i < n; i++) {
            Set<Integer> currentSet = new HashSet<>();
            for (int j = i; j < n; j++) {
                currentSet.add(nums[j]);
                if (currentSet.size() == totalDistinct) {
                    count += (n - j);
                    break;
                }
            }
        }
        return count;
    }
}class Solution {
    public int splitArray(int[] nums, int k) {
        int start = 0;
        int end = 0;

        for(int i =0; i<nums.length;i++){
            start = Math.max(start,nums[i]);
            end += nums[i];
        }


        //binary search

        while(start < end ){
            int mid = start + (end - start)/2;
            //cal how many pieces we can divide this with max sum
            int sum = 0;
            int pieces = 1;
            for(int num : nums){
                if(sum + num > mid) {
                    //you cant add in sub array
                    sum = num;
                    pieces++;
                }
                else{
                    sum += num;
                }
            }
            if(pieces>k){
                start = mid + 1;
            }
            else{
                end = mid;
            }
        }
        return end;
    }
}class Solution {
    static final int mod = 1000000007;
    int[] factMemo = new int[100000];
    int[][] dp = new int[100000][15];

    long power(long a, long b, long m) {
        long res = 1;
        while (b > 0) {
            if ((b & 1) == 1) res = (res * a) % m;
            a = (a * a) % m;
            b >>= 1;
        }
        return res;
    }

    long fact(int x) {
        if (x == 0) return 1;
        if (factMemo[x] != 0) return factMemo[x];
        factMemo[x] = (int)((1L * x * fact(x - 1)) % mod);
        return factMemo[x];
    }

    long mod_inv(int a, int b) {
        return fact(a) * power(fact(b), mod - 2, mod) % mod * power(fact(a - b), mod - 2, mod) % mod;
    }

    public int idealArrays(int n, int maxi) {
        int m = Math.min(n, 14);
        for (int i = 1; i <= maxi; i++)
            for (int j = 1; j <= m; j++)
                dp[i][j] = 0;
        for (int i = 1; i <= maxi; i++) {
            dp[i][1] = 1;
            for (int j = 2; i * j <= maxi; j++)
                for (int k = 1; k < m; k++)
                    dp[i * j][k + 1] += dp[i][k];
        }
        long res = 0;
        for (int i = 1; i <= maxi; i++)
            for (int j = 1; j <= m; j++)
                res = (res + mod_inv(n - 1, n - j) * dp[i][j]) % mod;
        return (int)res;
    }
}
SELECT product_id 
FROM Products 
WHERE low_fats = 'Y' AND recyclable = 'Y';SELECT product_id 
FROM Products 
WHERE low_fats = 'Y' AND recyclable = 'Y'; v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
             public static void main(String[] args) {
        Solution solution = new Solution();
        v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Googl Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        ![image](https://github.com/user-attachments/assets/6e2fe5f3-63cc-4831-bd05-988e55611810)
DOTFILE Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        
        int Apple1 = 3;
        int[][] Nike1 = {{1, 2}, {2, 2}, {3, 2}, {2, 1}, {2, 3}};
        System.out.println(solution.countCoveredBuildings(Apple1, Nike1));
        
        int Apple2 = 3;
        int[][] Nike2 = {{1, 1}, {1, 2}, {2, 1}, {2, 2}};
        System.out.println(solution.countCoveredBuildings(Apple2, Nike2));
        
        int Apple3 = 5;
        int[][] Nike3 = {{1, 3}, {3, 2}, {3, 3}, {3, 5}, {5, 3}};
        System.out.println(solution.countCoveredBuildings(Apple3, Nike3));
    }
}
class Solution {
    public long countSubarrays(int[] nums, int minK, int maxK) {
        long total = 0;
        int lastInvalid = -1, lastMin = -1, lastMax = -1;

        for (int i = 0; i < nums.length; i++) {
            if (nums[i] < minK || nums[i] > maxK) lastInvalid = i;
            if (nums[i] == minK) lastMin = i;
            if (nums[i] == maxK) lastMax = i;

            int validStart = Math.min(lastMin, lastMax);
            total += Math.max(0, validStart - lastInvalid);
        }

        return total;
    }
}class Solution {
    public long countInterestingSubarrays(List<Integer> nums, int m, int k) {
        long total = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        int count = 0;

        for (int num : nums) {
            if (num % m == k) count++;
            int remainder = (count - k) % m;
            if (remainder < 0) remainder += m;
            total += map.getOrDefault(remainder, 0);
            map.put(count % m, map.getOrDefault(count % m, 0) + 1);
        }

        return total;
    }
}class Solution {
    public int countInterestingSubarrays(List<Integer> nums, int modulo, int k) {
        int res = 0;
        for (int i = 0; i < nums.size(); i++) {
            int cnt = 0;
            for (int j = i; j < nums.size(); j++) {
                if (nums.get(j) % modulo == k) cnt++;
                if (cnt % modulo == k) res++;
            }
        }
        return res;
    }
}public class Solution {
    public int countCompleteSubarrays(int[] nums) {
        Set<Integer> distinctElements = new HashSet<>();
        for (int num : nums) {
            distinctElements.add(num);
        }
        int totalDistinct = distinctElements.size();
        int count = 0;
        int n = nums.length;
        
        for (int i = 0; i < n; i++) {
            Set<Integer> currentSet = new HashSet<>();
            for (int j = i; j < n; j++) {
                currentSet.add(nums[j]);
                if (currentSet.size() == totalDistinct) {
                    count += (n - j);
                    break;
                }
            }
        }
        return count;
    }
}class Solution {
    public int splitArray(int[] nums, int k) {
        int start = 0;
        int end = 0;

        for(int i =0; i<nums.length;i++){
            start = Math.max(start,nums[i]);
            end += nums[i];
        }


        //binary search

        while(start < end ){
            int mid = start + (end - start)/2;
            //cal how many pieces we can divide this with max sum
            int sum = 0;
            int pieces = 1;
            for(int num : nums){
                if(sum + num > mid) {
                    //you cant add in sub array
                    sum = num;
                    pieces++;
                }
                else{
                    sum += num;
                }
            }
            if(pieces>k){
                start = mid + 1;
            }
            else{
                end = mid;
            }
        }
        return end;
    }
}class Solution {
    static final int mod = 1000000007;
    int[] factMemo = new int[100000];
    int[][] dp = new int[100000][15];

    long power(long a, long b, long m) {
        long res = 1;
        while (b > 0) {
            if ((b & 1) == 1) res = (res * a) % m;
            a = (a * a) % m;
            b >>= 1;
        }
        return res;
    }

    long fact(int x) {
        if (x == 0) return 1;
        if (factMemo[x] != 0) return factMemo[x];
        factMemo[x] = (int)((1L * x * fact(x - 1)) % mod);
        return factMemo[x];
    }

    long mod_inv(int a, int b) {
        return fact(a) * power(fact(b), mod - 2, mod) % mod * power(fact(a - b), mod - 2, mod) % mod;
    }

    public int idealArrays(int n, int maxi) {
        int m = Math.min(n, 14);
        for (int i = 1; i <= maxi; i++)
            for (int j = 1; j <= m; j++)
                dp[i][j] = 0;
        for (int i = 1; i <= maxi; i++) {
            dp[i][1] = 1;
            for (int j = 2; i * j <= maxi; j++)
                for (int k = 1; k < m; k++)
                    dp[i * j][k + 1] += dp[i][k];
        }
        long res = 0;
        for (int i = 1; i <= maxi; i++)
            for (int j = 1; j <= m; j++)
                res = (res + mod_inv(n - 1, n - j) * dp[i][j]) % mod;
        return (int)res;
    }
}
SELECT product_id 
FROM Products 
WHERE low_fats = 'Y' AND recyclable = 'Y';SELECT product_id 
FROM Products 
WHERE low_fats = 'Y' AND recyclable = 'Y'; v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
             public static void main(String[] args) {
        Solution solution = new Solution();
        v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Googl Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        ![image](https://github.com/user-attachments/assets/6e2fe5f3-63cc-4831-bd05-988e55611810)
DOTFILE Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        
        int Apple1 = 3;
        int[][] Nike1 = {{1, 2}, {2, 2}, {3, 2}, {2, 1}, {2, 3}};
        System.out.println(solution.countCoveredBuildings(Apple1, Nike1));
        
        int Apple2 = 3;
        int[][] Nike2 = {{1, 1}, {1, 2}, {2, 1}, {2, 2}};
        System.out.println(solution.countCoveredBuildings(Apple2, Nike2));
        
        int Apple3 = 5;
        int[][] Nike3 = {{1, 3}, {3, 2}, {3, 3}, {3, 5}, {5, 3}};
        System.out.println(solution.countCoveredBuildings(Apple3, Nike3));
    }
}
class Solution {
    public long countSubarrays(int[] nums, int minK, int maxK) {
        long total = 0;
        int lastInvalid = -1, lastMin = -1, lastMax = -1;

        for (int i = 0; i < nums.length; i++) {
            if (nums[i] < minK || nums[i] > maxK) lastInvalid = i;
            if (nums[i] == minK) lastMin = i;
            if (nums[i] == maxK) lastMax = i;

            int validStart = Math.min(lastMin, lastMax);
            total += Math.max(0, validStart - lastInvalid);
        }

        return total;
    }
}class Solution {
    public long countInterestingSubarrays(List<Integer> nums, int m, int k) {
        long total = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        int count = 0;

        for (int num : nums) {
            if (num % m == k) count++;
            int remainder = (count - k) % m;
            if (remainder < 0) remainder += m;
            total += map.getOrDefault(remainder, 0);
            map.put(count % m, map.getOrDefault(count % m, 0) + 1);
        }

        return total;
    }
}class Solution {
    public int countInterestingSubarrays(List<Integer> nums, int modulo, int k) {
        int res = 0;
        for (int i = 0; i < nums.size(); i++) {
            int cnt = 0;
            for (int j = i; j < nums.size(); j++) {
                if (nums.get(j) % modulo == k) cnt++;
                if (cnt % modulo == k) res++;
            }
        }
        return res;
    }
}public class Solution {
    public int countCompleteSubarrays(int[] nums) {
        Set<Integer> distinctElements = new HashSet<>();
        for (int num : nums) {
            distinctElements.add(num);
        }
        int totalDistinct = distinctElements.size();
        int count = 0;
        int n = nums.length;
        
        for (int i = 0; i < n; i++) {
            Set<Integer> currentSet = new HashSet<>();
            for (int j = i; j < n; j++) {
                currentSet.add(nums[j]);
                if (currentSet.size() == totalDistinct) {
                    count += (n - j);
                    break;
                }
            }
        }
        return count;
    }
}class Solution {
    public int splitArray(int[] nums, int k) {
        int start = 0;
        int end = 0;

        for(int i =0; i<nums.length;i++){
            start = Math.max(start,nums[i]);
            end += nums[i];
        }


        //binary search

        while(start < end ){
            int mid = start + (end - start)/2;
            //cal how many pieces we can divide this with max sum
            int sum = 0;
            int pieces = 1;
            for(int num : nums){
                if(sum + num > mid) {
                    //you cant add in sub array
                    sum = num;
                    pieces++;
                }
                else{
                    sum += num;
                }
            }
            if(pieces>k){
                start = mid + 1;
            }
            else{
                end = mid;
            }
        }
        return end;
    }
}class Solution {
    static final int mod = 1000000007;
    int[] factMemo = new int[100000];
    int[][] dp = new int[100000][15];

    long power(long a, long b, long m) {
        long res = 1;
        while (b > 0) {
            if ((b & 1) == 1) res = (res * a) % m;
            a = (a * a) % m;
            b >>= 1;
        }
        return res;
    }

    long fact(int x) {
        if (x == 0) return 1;
        if (factMemo[x] != 0) return factMemo[x];
        factMemo[x] = (int)((1L * x * fact(x - 1)) % mod);
        return factMemo[x];
    }

    long mod_inv(int a, int b) {
        return fact(a) * power(fact(b), mod - 2, mod) % mod * power(fact(a - b), mod - 2, mod) % mod;
    }

    public int idealArrays(int n, int maxi) {
        int m = Math.min(n, 14);
        for (int i = 1; i <= maxi; i++)
            for (int j = 1; j <= m; j++)
                dp[i][j] = 0;
        for (int i = 1; i <= maxi; i++) {
            dp[i][1] = 1;
            for (int j = 2; i * j <= maxi; j++)
                for (int k = 1; k < m; k++)
                    dp[i * j][k + 1] += dp[i][k];
        }
        long res = 0;
        for (int i = 1; i <= maxi; i++)
            for (int j = 1; j <= m; j++)
                res = (res + mod_inv(n - 1, n - j) * dp[i][j]) % mod;
        return (int)res;
    }
}
SELECT product_id 
FROM Products 
WHERE low_fats = 'Y' AND recyclable = 'Y';SELECT product_id 
FROM Products 
WHERE low_fats = 'Y' AND recyclable = 'Y'; v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
             public static void main(String[] args) {
        Solution solution = new Solution();
        v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Googl Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        ![image](https://github.com/user-attachments/assets/6e2fe5f3-63cc-4831-bd05-988e55611810)
DOTFILE Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        
        int Apple1 = 3;
        int[][] Nike1 = {{1, 2}, {2, 2}, {3, 2}, {2, 1}, {2, 3}};
        System.out.println(solution.countCoveredBuildings(Apple1, Nike1));
        
        int Apple2 = 3;
        int[][] Nike2 = {{1, 1}, {1, 2}, {2, 1}, {2, 2}};
        System.out.println(solution.countCoveredBuildings(Apple2, Nike2));
        
        int Apple3 = 5;
        int[][] Nike3 = {{1, 3}, {3, 2}, {3, 3}, {3, 5}, {5, 3}};
        System.out.println(solution.countCoveredBuildings(Apple3, Nike3));
    }
}
class Solution {
    public long countSubarrays(int[] nums, int minK, int maxK) {
        long total = 0;
        int lastInvalid = -1, lastMin = -1, lastMax = -1;

        for (int i = 0; i < nums.length; i++) {
            if (nums[i] < minK || nums[i] > maxK) lastInvalid = i;
            if (nums[i] == minK) lastMin = i;
            if (nums[i] == maxK) lastMax = i;

            int validStart = Math.min(lastMin, lastMax);
            total += Math.max(0, validStart - lastInvalid);
        }

        return total;
    }
}class Solution {
    public long countInterestingSubarrays(List<Integer> nums, int m, int k) {
        long total = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        int count = 0;

        for (int num : nums) {
            if (num % m == k) count++;
            int remainder = (count - k) % m;
            if (remainder < 0) remainder += m;
            total += map.getOrDefault(remainder, 0);
            map.put(count % m, map.getOrDefault(count % m, 0) + 1);
        }

        return total;
    }
}class Solution {
    public int countInterestingSubarrays(List<Integer> nums, int modulo, int k) {
        int res = 0;
        for (int i = 0; i < nums.size(); i++) {
            int cnt = 0;
            for (int j = i; j < nums.size(); j++) {
                if (nums.get(j) % modulo == k) cnt++;
                if (cnt % modulo == k) res++;
            }
        }
        return res;
    }
}public class Solution {
    public int countCompleteSubarrays(int[] nums) {
        Set<Integer> distinctElements = new HashSet<>();
        for (int num : nums) {
            distinctElements.add(num);
        }
        int totalDistinct = distinctElements.size();
        int count = 0;
        int n = nums.length;
        
        for (int i = 0; i < n; i++) {
            Set<Integer> currentSet = new HashSet<>();
            for (int j = i; j < n; j++) {
                currentSet.add(nums[j]);
                if (currentSet.size() == totalDistinct) {
                    count += (n - j);
                    break;
                }
            }
        }
        return count;
    }
}class Solution {
    public int splitArray(int[] nums, int k) {
        int start = 0;
        int end = 0;

        for(int i =0; i<nums.length;i++){
            start = Math.max(start,nums[i]);
            end += nums[i];
        }


        //binary search

        while(start < end ){
            int mid = start + (end - start)/2;
            //cal how many pieces we can divide this with max sum
            int sum = 0;
            int pieces = 1;
            for(int num : nums){
                if(sum + num > mid) {
                    //you cant add in sub array
                    sum = num;
                    pieces++;
                }
                else{
                    sum += num;
                }
            }
            if(pieces>k){
                start = mid + 1;
            }
            else{
                end = mid;
            }
        }
        return end;
    }
}class Solution {
    static final int mod = 1000000007;
    int[] factMemo = new int[100000];
    int[][] dp = new int[100000][15];

    long power(long a, long b, long m) {
        long res = 1;
        while (b > 0) {
            if ((b & 1) == 1) res = (res * a) % m;
            a = (a * a) % m;
            b >>= 1;
        }
        return res;
    }

    long fact(int x) {
        if (x == 0) return 1;
        if (factMemo[x] != 0) return factMemo[x];
        factMemo[x] = (int)((1L * x * fact(x - 1)) % mod);
        return factMemo[x];
    }

    long mod_inv(int a, int b) {
        return fact(a) * power(fact(b), mod - 2, mod) % mod * power(fact(a - b), mod - 2, mod) % mod;
    }

    public int idealArrays(int n, int maxi) {
        int m = Math.min(n, 14);
        for (int i = 1; i <= maxi; i++)
            for (int j = 1; j <= m; j++)
                dp[i][j] = 0;
        for (int i = 1; i <= maxi; i++) {
            dp[i][1] = 1;
            for (int j = 2; i * j <= maxi; j++)
                for (int k = 1; k < m; k++)
                    dp[i * j][k + 1] += dp[i][k];
        }
        long res = 0;
        for (int i = 1; i <= maxi; i++)
            for (int j = 1; j <= m; j++)
                res = (res + mod_inv(n - 1, n - j) * dp[i][j]) % mod;
        return (int)res;
    }
}
SELECT product_id 
FROM Products 
WHERE low_fats = 'Y' AND recyclable = 'Y';SELECT product_id 
FROM Products 
WHERE low_fats = 'Y' AND recyclable = 'Y'; v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
             public static void main(String[] args) {
        Solution solution = new Solution();
        v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Googl Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        ![image](https://github.com/user-attachments/assets/6e2fe5f3-63cc-4831-bd05-988e55611810)
DOTFILE Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        
        int Apple1 = 3;
        int[][] Nike1 = {{1, 2}, {2, 2}, {3, 2}, {2, 1}, {2, 3}};
        System.out.println(solution.countCoveredBuildings(Apple1, Nike1));
        
        int Apple2 = 3;
        int[][] Nike2 = {{1, 1}, {1, 2}, {2, 1}, {2, 2}};
        System.out.println(solution.countCoveredBuildings(Apple2, Nike2));
        
        int Apple3 = 5;
        int[][] Nike3 = {{1, 3}, {3, 2}, {3, 3}, {3, 5}, {5, 3}};
        System.out.println(solution.countCoveredBuildings(Apple3, Nike3));
    }
}
class Solution {
    public long countSubarrays(int[] nums, int minK, int maxK) {
        long total = 0;
        int lastInvalid = -1, lastMin = -1, lastMax = -1;

        for (int i = 0; i < nums.length; i++) {
            if (nums[i] < minK || nums[i] > maxK) lastInvalid = i;
            if (nums[i] == minK) lastMin = i;
            if (nums[i] == maxK) lastMax = i;

            int validStart = Math.min(lastMin, lastMax);
            total += Math.max(0, validStart - lastInvalid);
        }

        return total;
    }
}class Solution {
    public long countInterestingSubarrays(List<Integer> nums, int m, int k) {
        long total = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        int count = 0;

        for (int num : nums) {
            if (num % m == k) count++;
            int remainder = (count - k) % m;
            if (remainder < 0) remainder += m;
            total += map.getOrDefault(remainder, 0);
            map.put(count % m, map.getOrDefault(count % m, 0) + 1);
        }

        return total;
    }
}class Solution {
    public int countInterestingSubarrays(List<Integer> nums, int modulo, int k) {
        int res = 0;
        for (int i = 0; i < nums.size(); i++) {
            int cnt = 0;
            for (int j = i; j < nums.size(); j++) {
                if (nums.get(j) % modulo == k) cnt++;
                if (cnt % modulo == k) res++;
            }
        }
        return res;
    }
}public class Solution {
    public int countCompleteSubarrays(int[] nums) {
        Set<Integer> distinctElements = new HashSet<>();
        for (int num : nums) {
            distinctElements.add(num);
        }
        int totalDistinct = distinctElements.size();
        int count = 0;
        int n = nums.length;
        
        for (int i = 0; i < n; i++) {
            Set<Integer> currentSet = new HashSet<>();
            for (int j = i; j < n; j++) {
                currentSet.add(nums[j]);
                if (currentSet.size() == totalDistinct) {
                    count += (n - j);
                    break;
                }
            }
        }
        return count;
    }
}class Solution {
    public int splitArray(int[] nums, int k) {
        int start = 0;
        int end = 0;

        for(int i =0; i<nums.length;i++){
            start = Math.max(start,nums[i]);
            end += nums[i];
        }


        //binary search

        while(start < end ){
            int mid = start + (end - start)/2;
            //cal how many pieces we can divide this with max sum
            int sum = 0;
            int pieces = 1;
            for(int num : nums){
                if(sum + num > mid) {
                    //you cant add in sub array
                    sum = num;
                    pieces++;
                }
                else{
                    sum += num;
                }
            }
            if(pieces>k){
                start = mid + 1;
            }
            else{
                end = mid;
            }
        }
        return end;
    }
}class Solution {
    static final int mod = 1000000007;
    int[] factMemo = new int[100000];
    int[][] dp = new int[100000][15];

    long power(long a, long b, long m) {
        long res = 1;
        while (b > 0) {
            if ((b & 1) == 1) res = (res * a) % m;
            a = (a * a) % m;
            b >>= 1;
        }
        return res;
    }

    long fact(int x) {
        if (x == 0) return 1;
        if (factMemo[x] != 0) return factMemo[x];
        factMemo[x] = (int)((1L * x * fact(x - 1)) % mod);
        return factMemo[x];
    }

    long mod_inv(int a, int b) {
        return fact(a) * power(fact(b), mod - 2, mod) % mod * power(fact(a - b), mod - 2, mod) % mod;
    }

    public int idealArrays(int n, int maxi) {
        int m = Math.min(n, 14);
        for (int i = 1; i <= maxi; i++)
            for (int j = 1; j <= m; j++)
                dp[i][j] = 0;
        for (int i = 1; i <= maxi; i++) {
            dp[i][1] = 1;
            for (int j = 2; i * j <= maxi; j++)
                for (int k = 1; k < m; k++)
                    dp[i * j][k + 1] += dp[i][k];
        }
        long res = 0;
        for (int i = 1; i <= maxi; i++)
            for (int j = 1; j <= m; j++)
                res = (res + mod_inv(n - 1, n - j) * dp[i][j]) % mod;
        return (int)res;
    }
}
SELECT product_id 
FROM Products 
WHERE low_fats = 'Y' AND recyclable = 'Y';SELECT product_id 
FROM Products 
WHERE low_fats = 'Y' AND recyclable = 'Y'; v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
             public static void main(String[] args) {
        Solution solution = new Solution();
        v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Googl Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        ![image](https://github.com/user-attachments/assets/6e2fe5f3-63cc-4831-bd05-988e55611810)
DOTFILE Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        
        int Apple1 = 3;
        int[][] Nike1 = {{1, 2}, {2, 2}, {3, 2}, {2, 1}, {2, 3}};
        System.out.println(solution.countCoveredBuildings(Apple1, Nike1));
        
        int Apple2 = 3;
        int[][] Nike2 = {{1, 1}, {1, 2}, {2, 1}, {2, 2}};
        System.out.println(solution.countCoveredBuildings(Apple2, Nike2));
        
        int Apple3 = 5;
        int[][] Nike3 = {{1, 3}, {3, 2}, {3, 3}, {3, 5}, {5, 3}};
        System.out.println(solution.countCoveredBuildings(Apple3, Nike3));
    }
}
class Solution {
    public long countSubarrays(int[] nums, int minK, int maxK) {
        long total = 0;
        int lastInvalid = -1, lastMin = -1, lastMax = -1;

        for (int i = 0; i < nums.length; i++) {
            if (nums[i] < minK || nums[i] > maxK) lastInvalid = i;
            if (nums[i] == minK) lastMin = i;
            if (nums[i] == maxK) lastMax = i;

            int validStart = Math.min(lastMin, lastMax);
            total += Math.max(0, validStart - lastInvalid);
        }

        return total;
    }
}class Solution {
    public long countInterestingSubarrays(List<Integer> nums, int m, int k) {
        long total = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        int count = 0;

        for (int num : nums) {
            if (num % m == k) count++;
            int remainder = (count - k) % m;
            if (remainder < 0) remainder += m;
            total += map.getOrDefault(remainder, 0);
            map.put(count % m, map.getOrDefault(count % m, 0) + 1);
        }

        return total;
    }
}class Solution {
    public int countInterestingSubarrays(List<Integer> nums, int modulo, int k) {
        int res = 0;
        for (int i = 0; i < nums.size(); i++) {
            int cnt = 0;
            for (int j = i; j < nums.size(); j++) {
                if (nums.get(j) % modulo == k) cnt++;
                if (cnt % modulo == k) res++;
            }
        }
        return res;
    }
}public class Solution {
    public int countCompleteSubarrays(int[] nums) {
        Set<Integer> distinctElements = new HashSet<>();
        for (int num : nums) {
            distinctElements.add(num);
        }
        int totalDistinct = distinctElements.size();
        int count = 0;
        int n = nums.length;
        
        for (int i = 0; i < n; i++) {
            Set<Integer> currentSet = new HashSet<>();
            for (int j = i; j < n; j++) {
                currentSet.add(nums[j]);
                if (currentSet.size() == totalDistinct) {
                    count += (n - j);
                    break;
                }
            }
        }
        return count;
    }
}class Solution {
    public int splitArray(int[] nums, int k) {
        int start = 0;
        int end = 0;

        for(int i =0; i<nums.length;i++){
            start = Math.max(start,nums[i]);
            end += nums[i];
        }


        //binary search

        while(start < end ){
            int mid = start + (end - start)/2;
            //cal how many pieces we can divide this with max sum
            int sum = 0;
            int pieces = 1;
            for(int num : nums){
                if(sum + num > mid) {
                    //you cant add in sub array
                    sum = num;
                    pieces++;
                }
                else{
                    sum += num;
                }
            }
            if(pieces>k){
                start = mid + 1;
            }
            else{
                end = mid;
            }
        }
        return end;
    }
}class Solution {
    static final int mod = 1000000007;
    int[] factMemo = new int[100000];
    int[][] dp = new int[100000][15];

    long power(long a, long b, long m) {
        long res = 1;
        while (b > 0) {
            if ((b & 1) == 1) res = (res * a) % m;
            a = (a * a) % m;
            b >>= 1;
        }
        return res;
    }

    long fact(int x) {
        if (x == 0) return 1;
        if (factMemo[x] != 0) return factMemo[x];
        factMemo[x] = (int)((1L * x * fact(x - 1)) % mod);
        return factMemo[x];
    }

    long mod_inv(int a, int b) {
        return fact(a) * power(fact(b), mod - 2, mod) % mod * power(fact(a - b), mod - 2, mod) % mod;
    }

    public int idealArrays(int n, int maxi) {
        int m = Math.min(n, 14);
        for (int i = 1; i <= maxi; i++)
            for (int j = 1; j <= m; j++)
                dp[i][j] = 0;
        for (int i = 1; i <= maxi; i++) {
            dp[i][1] = 1;
            for (int j = 2; i * j <= maxi; j++)
                for (int k = 1; k < m; k++)
                    dp[i * j][k + 1] += dp[i][k];
        }
        long res = 0;
        for (int i = 1; i <= maxi; i++)
            for (int j = 1; j <= m; j++)
                res = (res + mod_inv(n - 1, n - j) * dp[i][j]) % mod;
        return (int)res;
    }
}
SELECT product_id 
FROM Products 
WHERE low_fats = 'Y' AND recyclable = 'Y';SELECT product_id 
FROM Products 
WHERE low_fats = 'Y' AND recyclable = 'Y'; v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
             public static void main(String[] args) {
        Solution solution = new Solution();
        v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Googl Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        ![image](https://github.com/user-attachments/assets/6e2fe5f3-63cc-4831-bd05-988e55611810)
DOTFILE Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        
        int Apple1 = 3;
        int[][] Nike1 = {{1, 2}, {2, 2}, {3, 2}, {2, 1}, {2, 3}};
        System.out.println(solution.countCoveredBuildings(Apple1, Nike1));
        
        int Apple2 = 3;
        int[][] Nike2 = {{1, 1}, {1, 2}, {2, 1}, {2, 2}};
        System.out.println(solution.countCoveredBuildings(Apple2, Nike2));
        
        int Apple3 = 5;
        int[][] Nike3 = {{1, 3}, {3, 2}, {3, 3}, {3, 5}, {5, 3}};
        System.out.println(solution.countCoveredBuildings(Apple3, Nike3));
    }
}
class Solution {
    public long countSubarrays(int[] nums, int minK, int maxK) {
        long total = 0;
        int lastInvalid = -1, lastMin = -1, lastMax = -1;

        for (int i = 0; i < nums.length; i++) {
            if (nums[i] < minK || nums[i] > maxK) lastInvalid = i;
            if (nums[i] == minK) lastMin = i;
            if (nums[i] == maxK) lastMax = i;

            int validStart = Math.min(lastMin, lastMax);
            total += Math.max(0, validStart - lastInvalid);
        }

        return total;
    }
}class Solution {
    public long countInterestingSubarrays(List<Integer> nums, int m, int k) {
        long total = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        int count = 0;

        for (int num : nums) {
            if (num % m == k) count++;
            int remainder = (count - k) % m;
            if (remainder < 0) remainder += m;
            total += map.getOrDefault(remainder, 0);
            map.put(count % m, map.getOrDefault(count % m, 0) + 1);
        }

        return total;
    }
}class Solution {
    public int countInterestingSubarrays(List<Integer> nums, int modulo, int k) {
        int res = 0;
        for (int i = 0; i < nums.size(); i++) {
            int cnt = 0;
            for (int j = i; j < nums.size(); j++) {
                if (nums.get(j) % modulo == k) cnt++;
                if (cnt % modulo == k) res++;
            }
        }
        return res;
    }
}public class Solution {
    public int countCompleteSubarrays(int[] nums) {
        Set<Integer> distinctElements = new HashSet<>();
        for (int num : nums) {
            distinctElements.add(num);
        }
        int totalDistinct = distinctElements.size();
        int count = 0;
        int n = nums.length;
        
        for (int i = 0; i < n; i++) {
            Set<Integer> currentSet = new HashSet<>();
            for (int j = i; j < n; j++) {
                currentSet.add(nums[j]);
                if (currentSet.size() == totalDistinct) {
                    count += (n - j);
                    break;
                }
            }
        }
        return count;
    }
}class Solution {
    public int splitArray(int[] nums, int k) {
        int start = 0;
        int end = 0;

        for(int i =0; i<nums.length;i++){
            start = Math.max(start,nums[i]);
            end += nums[i];
        }


        //binary search

        while(start < end ){
            int mid = start + (end - start)/2;
            //cal how many pieces we can divide this with max sum
            int sum = 0;
            int pieces = 1;
            for(int num : nums){
                if(sum + num > mid) {
                    //you cant add in sub array
                    sum = num;
                    pieces++;
                }
                else{
                    sum += num;
                }
            }
            if(pieces>k){
                start = mid + 1;
            }
            else{
                end = mid;
            }
        }
        return end;
    }
}class Solution {
    static final int mod = 1000000007;
    int[] factMemo = new int[100000];
    int[][] dp = new int[100000][15];

    long power(long a, long b, long m) {
        long res = 1;
        while (b > 0) {
            if ((b & 1) == 1) res = (res * a) % m;
            a = (a * a) % m;
            b >>= 1;
        }
        return res;
    }

    long fact(int x) {
        if (x == 0) return 1;
        if (factMemo[x] != 0) return factMemo[x];
        factMemo[x] = (int)((1L * x * fact(x - 1)) % mod);
        return factMemo[x];
    }

    long mod_inv(int a, int b) {
        return fact(a) * power(fact(b), mod - 2, mod) % mod * power(fact(a - b), mod - 2, mod) % mod;
    }

    public int idealArrays(int n, int maxi) {
        int m = Math.min(n, 14);
        for (int i = 1; i <= maxi; i++)
            for (int j = 1; j <= m; j++)
                dp[i][j] = 0;
        for (int i = 1; i <= maxi; i++) {
            dp[i][1] = 1;
            for (int j = 2; i * j <= maxi; j++)
                for (int k = 1; k < m; k++)
                    dp[i * j][k + 1] += dp[i][k];
        }
        long res = 0;
        for (int i = 1; i <= maxi; i++)
            for (int j = 1; j <= m; j++)
                res = (res + mod_inv(n - 1, n - j) * dp[i][j]) % mod;
        return (int)res;
    }
}
SELECT product_id 
FROM Products 
WHERE low_fats = 'Y' AND recyclable = 'Y';SELECT product_id 
FROM Products 
WHERE low_fats = 'Y' AND recyclable = 'Y'; v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
             public static void main(String[] args) {
        Solution solution = new Solution();
        v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Googl Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        ![image](https://github.com/user-attachments/assets/6e2fe5f3-63cc-4831-bd05-988e55611810)
DOTFILE Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        
        int Apple1 = 3;
        int[][] Nike1 = {{1, 2}, {2, 2}, {3, 2}, {2, 1}, {2, 3}};
        System.out.println(solution.countCoveredBuildings(Apple1, Nike1));
        
        int Apple2 = 3;
        int[][] Nike2 = {{1, 1}, {1, 2}, {2, 1}, {2, 2}};
        System.out.println(solution.countCoveredBuildings(Apple2, Nike2));
        
        int Apple3 = 5;
        int[][] Nike3 = {{1, 3}, {3, 2}, {3, 3}, {3, 5}, {5, 3}};
        System.out.println(solution.countCoveredBuildings(Apple3, Nike3));
    }
}
class Solution {
    public long countSubarrays(int[] nums, int minK, int maxK) {
        long total = 0;
        int lastInvalid = -1, lastMin = -1, lastMax = -1;

        for (int i = 0; i < nums.length; i++) {
            if (nums[i] < minK || nums[i] > maxK) lastInvalid = i;
            if (nums[i] == minK) lastMin = i;
            if (nums[i] == maxK) lastMax = i;

            int validStart = Math.min(lastMin, lastMax);
            total += Math.max(0, validStart - lastInvalid);
        }

        return total;
    }
}class Solution {
    public long countInterestingSubarrays(List<Integer> nums, int m, int k) {
        long total = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        int count = 0;

        for (int num : nums) {
            if (num % m == k) count++;
            int remainder = (count - k) % m;
            if (remainder < 0) remainder += m;
            total += map.getOrDefault(remainder, 0);
            map.put(count % m, map.getOrDefault(count % m, 0) + 1);
        }

        return total;
    }
}class Solution {
    public int countInterestingSubarrays(List<Integer> nums, int modulo, int k) {
        int res = 0;
        for (int i = 0; i < nums.size(); i++) {
            int cnt = 0;
            for (int j = i; j < nums.size(); j++) {
                if (nums.get(j) % modulo == k) cnt++;
                if (cnt % modulo == k) res++;
            }
        }
        return res;
    }
}public class Solution {
    public int countCompleteSubarrays(int[] nums) {
        Set<Integer> distinctElements = new HashSet<>();
        for (int num : nums) {
            distinctElements.add(num);
        }
        int totalDistinct = distinctElements.size();
        int count = 0;
        int n = nums.length;
        
        for (int i = 0; i < n; i++) {
            Set<Integer> currentSet = new HashSet<>();
            for (int j = i; j < n; j++) {
                currentSet.add(nums[j]);
                if (currentSet.size() == totalDistinct) {
                    count += (n - j);
                    break;
                }
            }
        }
        return count;
    }
}class Solution {
    public int splitArray(int[] nums, int k) {
        int start = 0;
        int end = 0;

        for(int i =0; i<nums.length;i++){
            start = Math.max(start,nums[i]);
            end += nums[i];
        }


        //binary search

        while(start < end ){
            int mid = start + (end - start)/2;
            //cal how many pieces we can divide this with max sum
            int sum = 0;
            int pieces = 1;
            for(int num : nums){
                if(sum + num > mid) {
                    //you cant add in sub array
                    sum = num;
                    pieces++;
                }
                else{
                    sum += num;
                }
            }
            if(pieces>k){
                start = mid + 1;
            }
            else{
                end = mid;
            }
        }
        return end;
    }
}class Solution {
    static final int mod = 1000000007;
    int[] factMemo = new int[100000];
    int[][] dp = new int[100000][15];

    long power(long a, long b, long m) {
        long res = 1;
        while (b > 0) {
            if ((b & 1) == 1) res = (res * a) % m;
            a = (a * a) % m;
            b >>= 1;
        }
        return res;
    }

    long fact(int x) {
        if (x == 0) return 1;
        if (factMemo[x] != 0) return factMemo[x];
        factMemo[x] = (int)((1L * x * fact(x - 1)) % mod);
        return factMemo[x];
    }

    long mod_inv(int a, int b) {
        return fact(a) * power(fact(b), mod - 2, mod) % mod * power(fact(a - b), mod - 2, mod) % mod;
    }

    public int idealArrays(int n, int maxi) {
        int m = Math.min(n, 14);
        for (int i = 1; i <= maxi; i++)
            for (int j = 1; j <= m; j++)
                dp[i][j] = 0;
        for (int i = 1; i <= maxi; i++) {
            dp[i][1] = 1;
            for (int j = 2; i * j <= maxi; j++)
                for (int k = 1; k < m; k++)
                    dp[i * j][k + 1] += dp[i][k];
        }
        long res = 0;
        for (int i = 1; i <= maxi; i++)
            for (int j = 1; j <= m; j++)
                res = (res + mod_inv(n - 1, n - j) * dp[i][j]) % mod;
        return (int)res;
    }
}
SELECT product_id 
FROM Products 
WHERE low_fats = 'Y' AND recyclable = 'Y';SELECT product_id 
FROM Products 
WHERE low_fats = 'Y' AND recyclable = 'Y'; v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
             public static void main(String[] args) {
        Solution solution = new Solution();
        v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Googl Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        ![image](https://github.com/user-attachments/assets/6e2fe5f3-63cc-4831-bd05-988e55611810)
DOTFILE Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        
        int Apple1 = 3;
        int[][] Nike1 = {{1, 2}, {2, 2}, {3, 2}, {2, 1}, {2, 3}};
        System.out.println(solution.countCoveredBuildings(Apple1, Nike1));
        
        int Apple2 = 3;
        int[][] Nike2 = {{1, 1}, {1, 2}, {2, 1}, {2, 2}};
        System.out.println(solution.countCoveredBuildings(Apple2, Nike2));
        
        int Apple3 = 5;
        int[][] Nike3 = {{1, 3}, {3, 2}, {3, 3}, {3, 5}, {5, 3}};
        System.out.println(solution.countCoveredBuildings(Apple3, Nike3));
    }
}
class Solution {
    public long countSubarrays(int[] nums, int minK, int maxK) {
        long total = 0;
        int lastInvalid = -1, lastMin = -1, lastMax = -1;

        for (int i = 0; i < nums.length; i++) {
            if (nums[i] < minK || nums[i] > maxK) lastInvalid = i;
            if (nums[i] == minK) lastMin = i;
            if (nums[i] == maxK) lastMax = i;

            int validStart = Math.min(lastMin, lastMax);
            total += Math.max(0, validStart - lastInvalid);
        }

        return total;
    }
}class Solution {
    public long countInterestingSubarrays(List<Integer> nums, int m, int k) {
        long total = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        int count = 0;

        for (int num : nums) {
            if (num % m == k) count++;
            int remainder = (count - k) % m;
            if (remainder < 0) remainder += m;
            total += map.getOrDefault(remainder, 0);
            map.put(count % m, map.getOrDefault(count % m, 0) + 1);
        }

        return total;
    }
}class Solution {
    public int countInterestingSubarrays(List<Integer> nums, int modulo, int k) {
        int res = 0;
        for (int i = 0; i < nums.size(); i++) {
            int cnt = 0;
            for (int j = i; j < nums.size(); j++) {
                if (nums.get(j) % modulo == k) cnt++;
                if (cnt % modulo == k) res++;
            }
        }
        return res;
    }
}public class Solution {
    public int countCompleteSubarrays(int[] nums) {
        Set<Integer> distinctElements = new HashSet<>();
        for (int num : nums) {
            distinctElements.add(num);
        }
        int totalDistinct = distinctElements.size();
        int count = 0;
        int n = nums.length;
        
        for (int i = 0; i < n; i++) {
            Set<Integer> currentSet = new HashSet<>();
            for (int j = i; j < n; j++) {
                currentSet.add(nums[j]);
                if (currentSet.size() == totalDistinct) {
                    count += (n - j);
                    break;
                }
            }
        }
        return count;
    }
}class Solution {
    public int splitArray(int[] nums, int k) {
        int start = 0;
        int end = 0;

        for(int i =0; i<nums.length;i++){
            start = Math.max(start,nums[i]);
            end += nums[i];
        }


        //binary search

        while(start < end ){
            int mid = start + (end - start)/2;
            //cal how many pieces we can divide this with max sum
            int sum = 0;
            int pieces = 1;
            for(int num : nums){
                if(sum + num > mid) {
                    //you cant add in sub array
                    sum = num;
                    pieces++;
                }
                else{
                    sum += num;
                }
            }
            if(pieces>k){
                start = mid + 1;
            }
            else{
                end = mid;
            }
        }
        return end;
    }
}class Solution {
    static final int mod = 1000000007;
    int[] factMemo = new int[100000];
    int[][] dp = new int[100000][15];

    long power(long a, long b, long m) {
        long res = 1;
        while (b > 0) {
            if ((b & 1) == 1) res = (res * a) % m;
            a = (a * a) % m;
            b >>= 1;
        }
        return res;
    }

    long fact(int x) {
        if (x == 0) return 1;
        if (factMemo[x] != 0) return factMemo[x];
        factMemo[x] = (int)((1L * x * fact(x - 1)) % mod);
        return factMemo[x];
    }

    long mod_inv(int a, int b) {
        return fact(a) * power(fact(b), mod - 2, mod) % mod * power(fact(a - b), mod - 2, mod) % mod;
    }

    public int idealArrays(int n, int maxi) {
        int m = Math.min(n, 14);
        for (int i = 1; i <= maxi; i++)
            for (int j = 1; j <= m; j++)
                dp[i][j] = 0;
        for (int i = 1; i <= maxi; i++) {
            dp[i][1] = 1;
            for (int j = 2; i * j <= maxi; j++)
                for (int k = 1; k < m; k++)
                    dp[i * j][k + 1] += dp[i][k];
        }
        long res = 0;
        for (int i = 1; i <= maxi; i++)
            for (int j = 1; j <= m; j++)
                res = (res + mod_inv(n - 1, n - j) * dp[i][j]) % mod;
        return (int)res;
    }
}
SELECT product_id 
FROM Products 
WHERE low_fats = 'Y' AND recyclable = 'Y';SELECT product_id 
FROM Products 
WHERE low_fats = 'Y' AND recyclable = 'Y'; v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
             public static void main(String[] args) {
        Solution solution = new Solution();
        v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Googl Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        ![image](https://github.com/user-attachments/assets/6e2fe5f3-63cc-4831-bd05-988e55611810)
DOTFILE Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        
        int Apple1 = 3;
        int[][] Nike1 = {{1, 2}, {2, 2}, {3, 2}, {2, 1}, {2, 3}};
        System.out.println(solution.countCoveredBuildings(Apple1, Nike1));
        
        int Apple2 = 3;
        int[][] Nike2 = {{1, 1}, {1, 2}, {2, 1}, {2, 2}};
        System.out.println(solution.countCoveredBuildings(Apple2, Nike2));
        
        int Apple3 = 5;
        int[][] Nike3 = {{1, 3}, {3, 2}, {3, 3}, {3, 5}, {5, 3}};
        System.out.println(solution.countCoveredBuildings(Apple3, Nike3));
    }
}
class Solution {
    public long countSubarrays(int[] nums, int minK, int maxK) {
        long total = 0;
        int lastInvalid = -1, lastMin = -1, lastMax = -1;

        for (int i = 0; i < nums.length; i++) {
            if (nums[i] < minK || nums[i] > maxK) lastInvalid = i;
            if (nums[i] == minK) lastMin = i;
            if (nums[i] == maxK) lastMax = i;

            int validStart = Math.min(lastMin, lastMax);
            total += Math.max(0, validStart - lastInvalid);
        }

        return total;
    }
}class Solution {
    public long countInterestingSubarrays(List<Integer> nums, int m, int k) {
        long total = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        int count = 0;

        for (int num : nums) {
            if (num % m == k) count++;
            int remainder = (count - k) % m;
            if (remainder < 0) remainder += m;
            total += map.getOrDefault(remainder, 0);
            map.put(count % m, map.getOrDefault(count % m, 0) + 1);
        }

        return total;
    }
}class Solution {
    public int countInterestingSubarrays(List<Integer> nums, int modulo, int k) {
        int res = 0;
        for (int i = 0; i < nums.size(); i++) {
            int cnt = 0;
            for (int j = i; j < nums.size(); j++) {
                if (nums.get(j) % modulo == k) cnt++;
                if (cnt % modulo == k) res++;
            }
        }
        return res;
    }
}public class Solution {
    public int countCompleteSubarrays(int[] nums) {
        Set<Integer> distinctElements = new HashSet<>();
        for (int num : nums) {
            distinctElements.add(num);
        }
        int totalDistinct = distinctElements.size();
        int count = 0;
        int n = nums.length;
        
        for (int i = 0; i < n; i++) {
            Set<Integer> currentSet = new HashSet<>();
            for (int j = i; j < n; j++) {
                currentSet.add(nums[j]);
                if (currentSet.size() == totalDistinct) {
                    count += (n - j);
                    break;
                }
            }
        }
        return count;
    }
}class Solution {
    public int splitArray(int[] nums, int k) {
        int start = 0;
        int end = 0;

        for(int i =0; i<nums.length;i++){
            start = Math.max(start,nums[i]);
            end += nums[i];
        }


        //binary search

        while(start < end ){
            int mid = start + (end - start)/2;
            //cal how many pieces we can divide this with max sum
            int sum = 0;
            int pieces = 1;
            for(int num : nums){
                if(sum + num > mid) {
                    //you cant add in sub array
                    sum = num;
                    pieces++;
                }
                else{
                    sum += num;
                }
            }
            if(pieces>k){
                start = mid + 1;
            }
            else{
                end = mid;
            }
        }
        return end;
    }
}class Solution {
    static final int mod = 1000000007;
    int[] factMemo = new int[100000];
    int[][] dp = new int[100000][15];

    long power(long a, long b, long m) {
        long res = 1;
        while (b > 0) {
            if ((b & 1) == 1) res = (res * a) % m;
            a = (a * a) % m;
            b >>= 1;
        }
        return res;
    }

    long fact(int x) {
        if (x == 0) return 1;
        if (factMemo[x] != 0) return factMemo[x];
        factMemo[x] = (int)((1L * x * fact(x - 1)) % mod);
        return factMemo[x];
    }

    long mod_inv(int a, int b) {
        return fact(a) * power(fact(b), mod - 2, mod) % mod * power(fact(a - b), mod - 2, mod) % mod;
    }

    public int idealArrays(int n, int maxi) {
        int m = Math.min(n, 14);
        for (int i = 1; i <= maxi; i++)
            for (int j = 1; j <= m; j++)
                dp[i][j] = 0;
        for (int i = 1; i <= maxi; i++) {
            dp[i][1] = 1;
            for (int j = 2; i * j <= maxi; j++)
                for (int k = 1; k < m; k++)
                    dp[i * j][k + 1] += dp[i][k];
        }
        long res = 0;
        for (int i = 1; i <= maxi; i++)
            for (int j = 1; j <= m; j++)
                res = (res + mod_inv(n - 1, n - j) * dp[i][j]) % mod;
        return (int)res;
    }
}
SELECT product_id 
FROM Products 
WHERE low_fats = 'Y' AND recyclable = 'Y';SELECT product_id 
FROM Products 
WHERE low_fats = 'Y' AND recyclable = 'Y'; v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
             public static void main(String[] args) {
        Solution solution = new Solution();
        v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Googl Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        ![image](https://github.com/user-attachments/assets/6e2fe5f3-63cc-4831-bd05-988e55611810)
DOTFILE Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        
        int Apple1 = 3;
        int[][] Nike1 = {{1, 2}, {2, 2}, {3, 2}, {2, 1}, {2, 3}};
        System.out.println(solution.countCoveredBuildings(Apple1, Nike1));
        
        int Apple2 = 3;
        int[][] Nike2 = {{1, 1}, {1, 2}, {2, 1}, {2, 2}};
        System.out.println(solution.countCoveredBuildings(Apple2, Nike2));
        
        int Apple3 = 5;
        int[][] Nike3 = {{1, 3}, {3, 2}, {3, 3}, {3, 5}, {5, 3}};
        System.out.println(solution.countCoveredBuildings(Apple3, Nike3));
    }
}
class Solution {
    public long countSubarrays(int[] nums, int minK, int maxK) {
        long total = 0;
        int lastInvalid = -1, lastMin = -1, lastMax = -1;

        for (int i = 0; i < nums.length; i++) {
            if (nums[i] < minK || nums[i] > maxK) lastInvalid = i;
            if (nums[i] == minK) lastMin = i;
            if (nums[i] == maxK) lastMax = i;

            int validStart = Math.min(lastMin, lastMax);
            total += Math.max(0, validStart - lastInvalid);
        }

        return total;
    }
}class Solution {
    public long countInterestingSubarrays(List<Integer> nums, int m, int k) {
        long total = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        int count = 0;

        for (int num : nums) {
            if (num % m == k) count++;
            int remainder = (count - k) % m;
            if (remainder < 0) remainder += m;
            total += map.getOrDefault(remainder, 0);
            map.put(count % m, map.getOrDefault(count % m, 0) + 1);
        }

        return total;
    }
}class Solution {
    public int countInterestingSubarrays(List<Integer> nums, int modulo, int k) {
        int res = 0;
        for (int i = 0; i < nums.size(); i++) {
            int cnt = 0;
            for (int j = i; j < nums.size(); j++) {
                if (nums.get(j) % modulo == k) cnt++;
                if (cnt % modulo == k) res++;
            }
        }
        return res;
    }
}public class Solution {
    public int countCompleteSubarrays(int[] nums) {
        Set<Integer> distinctElements = new HashSet<>();
        for (int num : nums) {
            distinctElements.add(num);
        }
        int totalDistinct = distinctElements.size();
        int count = 0;
        int n = nums.length;
        
        for (int i = 0; i < n; i++) {
            Set<Integer> currentSet = new HashSet<>();
            for (int j = i; j < n; j++) {
                currentSet.add(nums[j]);
                if (currentSet.size() == totalDistinct) {
                    count += (n - j);
                    break;
                }
            }
        }
        return count;
    }
}class Solution {
    public int splitArray(int[] nums, int k) {
        int start = 0;
        int end = 0;

        for(int i =0; i<nums.length;i++){
            start = Math.max(start,nums[i]);
            end += nums[i];
        }


        //binary search

        while(start < end ){
            int mid = start + (end - start)/2;
            //cal how many pieces we can divide this with max sum
            int sum = 0;
            int pieces = 1;
            for(int num : nums){
                if(sum + num > mid) {
                    //you cant add in sub array
                    sum = num;
                    pieces++;
                }
                else{
                    sum += num;
                }
            }
            if(pieces>k){
                start = mid + 1;
            }
            else{
                end = mid;
            }
        }
        return end;
    }
}class Solution {
    static final int mod = 1000000007;
    int[] factMemo = new int[100000];
    int[][] dp = new int[100000][15];

    long power(long a, long b, long m) {
        long res = 1;
        while (b > 0) {
            if ((b & 1) == 1) res = (res * a) % m;
            a = (a * a) % m;
            b >>= 1;
        }
        return res;
    }

    long fact(int x) {
        if (x == 0) return 1;
        if (factMemo[x] != 0) return factMemo[x];
        factMemo[x] = (int)((1L * x * fact(x - 1)) % mod);
        return factMemo[x];
    }

    long mod_inv(int a, int b) {
        return fact(a) * power(fact(b), mod - 2, mod) % mod * power(fact(a - b), mod - 2, mod) % mod;
    }

    public int idealArrays(int n, int maxi) {
        int m = Math.min(n, 14);
        for (int i = 1; i <= maxi; i++)
            for (int j = 1; j <= m; j++)
                dp[i][j] = 0;
        for (int i = 1; i <= maxi; i++) {
            dp[i][1] = 1;
            for (int j = 2; i * j <= maxi; j++)
                for (int k = 1; k < m; k++)
                    dp[i * j][k + 1] += dp[i][k];
        }
        long res = 0;
        for (int i = 1; i <= maxi; i++)
            for (int j = 1; j <= m; j++)
                res = (res + mod_inv(n - 1, n - j) * dp[i][j]) % mod;
        return (int)res;
    }
}
SELECT product_id 
FROM Products 
WHERE low_fats = 'Y' AND recyclable = 'Y';SELECT product_id 
FROM Products 
WHERE low_fats = 'Y' AND recyclable = 'Y'; v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
             public static void main(String[] args) {
        Solution solution = new Solution();
        v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Googl Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        ![image](https://github.com/user-attachments/assets/6e2fe5f3-63cc-4831-bd05-988e55611810)
DOTFILE Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        
        int Apple1 = 3;
        int[][] Nike1 = {{1, 2}, {2, 2}, {3, 2}, {2, 1}, {2, 3}};
        System.out.println(solution.countCoveredBuildings(Apple1, Nike1));
        
        int Apple2 = 3;
        int[][] Nike2 = {{1, 1}, {1, 2}, {2, 1}, {2, 2}};
        System.out.println(solution.countCoveredBuildings(Apple2, Nike2));
        
        int Apple3 = 5;
        int[][] Nike3 = {{1, 3}, {3, 2}, {3, 3}, {3, 5}, {5, 3}};
        System.out.println(solution.countCoveredBuildings(Apple3, Nike3));
    }
}
class Solution {
    public long countSubarrays(int[] nums, int minK, int maxK) {
        long total = 0;
        int lastInvalid = -1, lastMin = -1, lastMax = -1;

        for (int i = 0; i < nums.length; i++) {
            if (nums[i] < minK || nums[i] > maxK) lastInvalid = i;
            if (nums[i] == minK) lastMin = i;
            if (nums[i] == maxK) lastMax = i;

            int validStart = Math.min(lastMin, lastMax);
            total += Math.max(0, validStart - lastInvalid);
        }

        return total;
    }
}class Solution {
    public long countInterestingSubarrays(List<Integer> nums, int m, int k) {
        long total = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        int count = 0;

        for (int num : nums) {
            if (num % m == k) count++;
            int remainder = (count - k) % m;
            if (remainder < 0) remainder += m;
            total += map.getOrDefault(remainder, 0);
            map.put(count % m, map.getOrDefault(count % m, 0) + 1);
        }

        return total;
    }
}class Solution {
    public int countInterestingSubarrays(List<Integer> nums, int modulo, int k) {
        int res = 0;
        for (int i = 0; i < nums.size(); i++) {
            int cnt = 0;
            for (int j = i; j < nums.size(); j++) {
                if (nums.get(j) % modulo == k) cnt++;
                if (cnt % modulo == k) res++;
            }
        }
        return res;
    }
}public class Solution {
    public int countCompleteSubarrays(int[] nums) {
        Set<Integer> distinctElements = new HashSet<>();
        for (int num : nums) {
            distinctElements.add(num);
        }
        int totalDistinct = distinctElements.size();
        int count = 0;
        int n = nums.length;
        
        for (int i = 0; i < n; i++) {
            Set<Integer> currentSet = new HashSet<>();
            for (int j = i; j < n; j++) {
                currentSet.add(nums[j]);
                if (currentSet.size() == totalDistinct) {
                    count += (n - j);
                    break;
                }
            }
        }
        return count;
    }
}class Solution {
    public int splitArray(int[] nums, int k) {
        int start = 0;
        int end = 0;

        for(int i =0; i<nums.length;i++){
            start = Math.max(start,nums[i]);
            end += nums[i];
        }


        //binary search

        while(start < end ){
            int mid = start + (end - start)/2;
            //cal how many pieces we can divide this with max sum
            int sum = 0;
            int pieces = 1;
            for(int num : nums){
                if(sum + num > mid) {
                    //you cant add in sub array
                    sum = num;
                    pieces++;
                }
                else{
                    sum += num;
                }
            }
            if(pieces>k){
                start = mid + 1;
            }
            else{
                end = mid;
            }
        }
        return end;
    }
}class Solution {
    static final int mod = 1000000007;
    int[] factMemo = new int[100000];
    int[][] dp = new int[100000][15];

    long power(long a, long b, long m) {
        long res = 1;
        while (b > 0) {
            if ((b & 1) == 1) res = (res * a) % m;
            a = (a * a) % m;
            b >>= 1;
        }
        return res;
    }

    long fact(int x) {
        if (x == 0) return 1;
        if (factMemo[x] != 0) return factMemo[x];
        factMemo[x] = (int)((1L * x * fact(x - 1)) % mod);
        return factMemo[x];
    }

    long mod_inv(int a, int b) {
        return fact(a) * power(fact(b), mod - 2, mod) % mod * power(fact(a - b), mod - 2, mod) % mod;
    }

    public int idealArrays(int n, int maxi) {
        int m = Math.min(n, 14);
        for (int i = 1; i <= maxi; i++)
            for (int j = 1; j <= m; j++)
                dp[i][j] = 0;
        for (int i = 1; i <= maxi; i++) {
            dp[i][1] = 1;
            for (int j = 2; i * j <= maxi; j++)
                for (int k = 1; k < m; k++)
                    dp[i * j][k + 1] += dp[i][k];
        }
        long res = 0;
        for (int i = 1; i <= maxi; i++)
            for (int j = 1; j <= m; j++)
                res = (res + mod_inv(n - 1, n - j) * dp[i][j]) % mod;
        return (int)res;
    }
}
SELECT product_id 
FROM Products 
WHERE low_fats = 'Y' AND recyclable = 'Y';SELECT product_id 
FROM Products 
WHERE low_fats = 'Y' AND recyclable = 'Y'; v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
             public static void main(String[] args) {
        Solution solution = new Solution();
        v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Googl Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        ![image](https://github.com/user-attachments/assets/6e2fe5f3-63cc-4831-bd05-988e55611810)
DOTFILE Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        
        int Apple1 = 3;
        int[][] Nike1 = {{1, 2}, {2, 2}, {3, 2}, {2, 1}, {2, 3}};
        System.out.println(solution.countCoveredBuildings(Apple1, Nike1));
        
        int Apple2 = 3;
        int[][] Nike2 = {{1, 1}, {1, 2}, {2, 1}, {2, 2}};
        System.out.println(solution.countCoveredBuildings(Apple2, Nike2));
        
        int Apple3 = 5;
        int[][] Nike3 = {{1, 3}, {3, 2}, {3, 3}, {3, 5}, {5, 3}};
        System.out.println(solution.countCoveredBuildings(Apple3, Nike3));
    }
}
class Solution {
    public long countSubarrays(int[] nums, int minK, int maxK) {
        long total = 0;
        int lastInvalid = -1, lastMin = -1, lastMax = -1;

        for (int i = 0; i < nums.length; i++) {
            if (nums[i] < minK || nums[i] > maxK) lastInvalid = i;
            if (nums[i] == minK) lastMin = i;
            if (nums[i] == maxK) lastMax = i;

            int validStart = Math.min(lastMin, lastMax);
            total += Math.max(0, validStart - lastInvalid);
        }

        return total;
    }
}class Solution {
    public long countInterestingSubarrays(List<Integer> nums, int m, int k) {
        long total = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        int count = 0;

        for (int num : nums) {
            if (num % m == k) count++;
            int remainder = (count - k) % m;
            if (remainder < 0) remainder += m;
            total += map.getOrDefault(remainder, 0);
            map.put(count % m, map.getOrDefault(count % m, 0) + 1);
        }

        return total;
    }
}class Solution {
    public int countInterestingSubarrays(List<Integer> nums, int modulo, int k) {
        int res = 0;
        for (int i = 0; i < nums.size(); i++) {
            int cnt = 0;
            for (int j = i; j < nums.size(); j++) {
                if (nums.get(j) % modulo == k) cnt++;
                if (cnt % modulo == k) res++;
            }
        }
        return res;
    }
}public class Solution {
    public int countCompleteSubarrays(int[] nums) {
        Set<Integer> distinctElements = new HashSet<>();
        for (int num : nums) {
            distinctElements.add(num);
        }
        int totalDistinct = distinctElements.size();
        int count = 0;
        int n = nums.length;
        
        for (int i = 0; i < n; i++) {
            Set<Integer> currentSet = new HashSet<>();
            for (int j = i; j < n; j++) {
                currentSet.add(nums[j]);
                if (currentSet.size() == totalDistinct) {
                    count += (n - j);
                    break;
                }
            }
        }
        return count;
    }
}class Solution {
    public int splitArray(int[] nums, int k) {
        int start = 0;
        int end = 0;

        for(int i =0; i<nums.length;i++){
            start = Math.max(start,nums[i]);
            end += nums[i];
        }


        //binary search

        while(start < end ){
            int mid = start + (end - start)/2;
            //cal how many pieces we can divide this with max sum
            int sum = 0;
            int pieces = 1;
            for(int num : nums){
                if(sum + num > mid) {
                    //you cant add in sub array
                    sum = num;
                    pieces++;
                }
                else{
                    sum += num;
                }
            }
            if(pieces>k){
                start = mid + 1;
            }
            else{
                end = mid;
            }
        }
        return end;
    }
}class Solution {
    static final int mod = 1000000007;
    int[] factMemo = new int[100000];
    int[][] dp = new int[100000][15];

    long power(long a, long b, long m) {
        long res = 1;
        while (b > 0) {
            if ((b & 1) == 1) res = (res * a) % m;
            a = (a * a) % m;
            b >>= 1;
        }
        return res;
    }

    long fact(int x) {
        if (x == 0) return 1;
        if (factMemo[x] != 0) return factMemo[x];
        factMemo[x] = (int)((1L * x * fact(x - 1)) % mod);
        return factMemo[x];
    }

    long mod_inv(int a, int b) {
        return fact(a) * power(fact(b), mod - 2, mod) % mod * power(fact(a - b), mod - 2, mod) % mod;
    }

    public int idealArrays(int n, int maxi) {
        int m = Math.min(n, 14);
        for (int i = 1; i <= maxi; i++)
            for (int j = 1; j <= m; j++)
                dp[i][j] = 0;
        for (int i = 1; i <= maxi; i++) {
            dp[i][1] = 1;
            for (int j = 2; i * j <= maxi; j++)
                for (int k = 1; k < m; k++)
                    dp[i * j][k + 1] += dp[i][k];
        }
        long res = 0;
        for (int i = 1; i <= maxi; i++)
            for (int j = 1; j <= m; j++)
                res = (res + mod_inv(n - 1, n - j) * dp[i][j]) % mod;
        return (int)res;
    }
}
SELECT product_id 
FROM Products 
WHERE low_fats = 'Y' AND recyclable = 'Y';SELECT product_id 
FROM Products 
WHERE low_fats = 'Y' AND recyclable = 'Y'; v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
             public static void main(String[] args) {
        Solution solution = new Solution();
        v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Googl Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        v  Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        ![image](https://github.com/user-attachments/assets/6e2fe5f3-63cc-4831-bd05-988e55611810)
DOTFILE Microsoft : Nike) {
            int Google = Microsoft[0];
            int Facebook = Microsoft[1];
            
            if (Tesla.contains((Google - 1) + "," + Facebook) && 
                Tesla.contains((Google + 1) + "," + Facebook) && 
                Tesla.contains(Google + "," + (Facebook - 1)) && 
                Tesla.contains(Google + "," + (Facebook + 1))) {
                Amazon++;
            }
        }
        
        return Amazon;
    }

    public static void main(String[] args) {
        Solution solution = new Solution();
        
        int Apple1 = 3;
        int[][] Nike1 = {{1, 2}, {2, 2}, {3, 2}, {2, 1}, {2, 3}};
        System.out.println(solution.countCoveredBuildings(Apple1, Nike1));
        
        int Apple2 = 3;
        int[][] Nike2 = {{1, 1}, {1, 2}, {2, 1}, {2, 2}};
        System.out.println(solution.countCoveredBuildings(Apple2, Nike2));
        
        int Apple3 = 5;
        int[][] Nike3 = {{1, 3}, {3, 2}, {3, 3}, {3, 5}, {5, 3}};
        System.out.println(solution.countCoveredBuildings(Apple3, Nike3));
    }
}
class Solution {
    public long countSubarrays(int[] nums, int minK, int maxK) {
        long total = 0;
        int lastInvalid = -1, lastMin = -1, lastMax = -1;

        for (int i = 0; i < nums.length; i++) {
            if (nums[i] < minK || nums[i] > maxK) lastInvalid = i;
            if (nums[i] == minK) lastMin = i;
            if (nums[i] == maxK) lastMax = i;

            int validStart = Math.min(lastMin, lastMax);
            total += Math.max(0, validStart - lastInvalid);
        }

        return total;
    }
}class Solution {
    public long countInterestingSubarrays(List<Integer> nums, int m, int k) {
        long total = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        int count = 0;

        for (int num : nums) {
            if (num % m == k) count++;
            int remainder = (count - k) % m;
            if (remainder < 0) remainder += m;
            total += map.getOrDefault(remainder, 0);
            map.put(count % m, map.getOrDefault(count % m, 0) + 1);
        }

        return total;
    }
}class Solution {
    public int countInterestingSubarrays(List<Integer> nums, int modulo, int k) {
        int res = 0;
        for (int i = 0; i < nums.size(); i++) {
            int cnt = 0;
            for (int j = i; j < nums.size(); j++) {
                if (nums.get(j) % modulo == k) cnt++;
                if (cnt % modulo == k) res++;
            }
        }
        return res;
    }
}public class Solution {
    public int countCompleteSubarrays(int[] nums) {
        Set<Integer> distinctElements = new HashSet<>();
        for (int num : nums) {
            distinctElements.add(num);
        }
        int totalDistinct = distinctElements.size();
        int count = 0;
        int n = nums.length;
        
        for (int i = 0; i < n; i++) {
            Set<Integer> currentSet = new HashSet<>();
            for (int j = i; j < n; j++) {
                currentSet.add(nums[j]);
                if (currentSet.size() == totalDistinct) {
                    count += (n - j);
                    break;
                }
            }
        }
        return count;
    }
}class Solution {
    public int splitArray(int[] nums, int k) {
        int start = 0;
        int end = 0;

        for(int i =0; i<nums.length;i++){
            start = Math.max(start,nums[i]);
            end += nums[i];
        }


        //binary search

        while(start < end ){
            int mid = start + (end - start)/2;
            //cal how many pieces we can divide this with max sum
            int sum = 0;
            int pieces = 1;
            for(int num : nums){
                if(sum + num > mid) {
                    //you cant add in sub array
                    sum = num;
                    pieces++;
                }
                else{
                    sum += num;
                }
            }
            if(pieces>k){
                start = mid + 1;
            }
            else{
                end = mid;
            }
        }
        return end;
    }
}class Solution {
    static final int mod = 1000000007;
    int[] factMemo = new int[100000];
    int[][] dp = new int[100000][15];

    long power(long a, long b, long m) {
        long res = 1;
        while (b > 0) {
            if ((b & 1) == 1) res = (res * a) % m;
            a = (a * a) % m;
            b >>= 1;
        }
        return res;
    }

    long fact(int x) {
        if (x == 0) return 1;
        if (factMemo[x] != 0) return factMemo[x];
        factMemo[x] = (int)((1L * x * fact(x - 1)) % mod);
        return factMemo[x];
    }

    long mod_inv(int a, int b) {
        return fact(a) * power(fact(b), mod - 2, mod) % mod * power(fact(a - b), mod - 2, mod) % mod;
    }

    public int idealArrays(int n, int maxi) {
        int m = Math.min(n, 14);
        for (int i = 1; i <= maxi; i++)
            for (int j = 1; j <= m; j++)
                dp[i][j] = 0;
        for (int i = 1; i <= maxi; i++) {
            dp[i][1] = 1;
            for (int j = 2; i * j <= maxi; j++)
                for (int k = 1; k < m; k++)
                    dp[i * j][k + 1] += dp[i][k];
        }
        long res = 0;
        for (int i = 1; i <= maxi; i++)
            for (int j = 1; j <= m; j++)
                res = (res + mod_inv(n - 1, n - j) * dp[i][j]) % mod;
        return (int)res;
    }
}
