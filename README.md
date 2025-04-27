  Microsoft : Nike) {
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
