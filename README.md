
 char replaceForMax = ' ';
        for (char c : maxChars) {
            if (c != '9') {
                replaceForMax = c;
                break;
            }
        }class Solution {
    public int minMaxDifference(int num) {
        String str = Integer.toString(num);
        char[] maxChars = str.toCharArray();
        char[] minChars = str.toCharArray();

        // Step 1: Find first non-9 digit
 char replaceForMax = ' ';
        for (char c : maxChars) {
            if (c != '9') {
                replaceForMax = c;
                break;
            }
        }class Solution {
    public int minMaxDifference(int num) {
        String str = Integer.toString(num);
        char[] maxChars = str.toCharArray();
        char[] minChars = str.toCharArray();

        // Step 1: Find first non-9 digit
 char replaceForMax = ' ';
        for (char c : maxChars) {
            if (c != '9') {
                replaceForMax = c;
                break;
            }
        }class Solution {
    public int minMaxDifference(int num) {
        String str = Integer.toString(num);
        char[] maxChars = str.toCharArray();
        char[] minChars = str.toCharArray();

        // Step 1: Find first non-9 digit for max replacement
        char replaceForMax = ' ';
        for (char c : maxChars) {
            if (c != '9') {
                replaceForMax = c;
                break;
            }
        }

        // Step 2: Replace that digit with 9 for max
        for (int i = 0; i < maxChars.length; i++) {
            if (maxChars[i] == replaceForMax) {
                maxChars[i] = '9';
            }
        }

        // Step 3: Replace first digit for min with 0
        char replaceForMin = minChars[0];
        for (int i = 0; i < minChars.length; i++) {
            if (minChars[i] == replaceForMin) {
                minChars[i] = '0';
            }
        }

        // Step 4: Convert and calculate result
        int maxVal = Integer.parseInt(new String(maxChars));
        int minVal = Integer.parseInt(new String(minChars));

        return maxVal - minVal;
    }
}![/user-attachments/assets/a31ac698-48d7-4d5a-9af8-dbcea7599f9b)
class Solution {
    private static final int MOD = (int) 1e8;
    private int maxDfromAtoB(int a, int b, int k, int n, int[][] freq) {
        int cnt = Integer.MIN_VALUE;
        int[][] minFreq = { {MOD, MOD}, {MOD, MOD} };
        int freqA = 0, freqB = 0;
        int prevA = 0, prevB = 0;
        int l = 0;
        for (int r = k - 1; r < n; r++) {
            freqA = freq[a][r + 1];
            freqB = freq[b][r + 1];
            while (r - l + 1 >= k && freqB - prevB >= 2) {
                minFreq[prevA & 1][prevB & 1] = Math.min(minFreq[prevA & 1][prevB & 1], prevA - prevB);
                prevA = freq[a][l + 1];
                prevB = freq[b][l + 1];
   class Solution {
    public int maxAdjacentDistance(int[] nums) {
        int n = nums.length;
        int maxa = Math.abs(nums[0] - nums[n - 1]);
        for (int i = 0; i < n - 1; i++) {
            maxa = Math.max(maxa, Math.abs(nums[i] - nums[i + 1]));
        }
        return maxa;
    }
}             l++;
            }
            cnt = Math.max(cnt, freqA - freqB - minFreq[1 - (freqA & 1)][freqB & 1]);
        }
        return cnt;
    }

    public int maxDifference(String s, int k) {
        int n = s.length();
        int[][] freq = new int[5][n + 1];
        for (int i = 0; i < n; i++) {
            for (int d = 0; d < 5; d++) {
                freq[d][i + 1] = freq[d][i];
            }
            freq[s.charAt(i) - '0'][i + 1]++;
        }
        int ans = Integer.MIN_VALUE;
        for (int a = 0; a < 5; a++) {
            if (freq[a][n] == 0)
                continue;
            for (int b = 0; b < 5; b++) {
                if (a == b || freq[b][n] == 0)
                    continue;
                ans = Math.max(ans, maxDfromAtoB(a, b, k, n, freq));
            }
        }
        return ans;
    }
}import java.util.*;

class Solution {
    public int maxDifference(String s) {
        Map<Character, Integer> freq = new HashMap<>();
        int minEven = Integer.MAX_VALUE;
        int maxOdd = Integer.MIN_VALUE;

        for (char ch : s.toCharArray()) {
            freq.put(ch, freq.getOrDefault(ch, 0) + 1);
        }

        for (int count : freq.values()) {
            if (count % 2 == 0) {
                minEven = Math.min(minEven, count);
            } else {
                maxOdd = Math.max(maxOdd, count);
            }
        }

        return maxOdd - minEven;
    }
}class Solution {
    public int findKthNumber(int n, int k) {
        long curr = 1;
        k -= 1; // we already include 1 in our result

        while (k > 0) {
            long count = getCount(curr, n);
            if (count <= k) {
                // skip current prefix subtree
                curr++;
                k -= count;
            } else {
                // go deeper in the tree
                curr *= 10;
                k -= 1;
            }
        }
        return (int) curr;
    }

    private long getCount(long prefix, long n) {
        long count = 0;
        long current = prefix;
        long next = prefix + 1;

        while (current <= n) {
            count += Math.min(n + 1, next) - current;
            current *= 10;
            next *= 10;
        }
        return count;
    }
}class Solution {

    public String clearStars(String s) {
        Deque<Integer>[] cnt = new Deque[26];
        for (int i = 0; i < 26; i++) {
            cnt[i] = new ArrayDeque<>();
        }
        char[] arr = s.toCharArray();
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] != '*') {
                cnt[arr[i] - 'a'].push(i);
            } else {
                for (int j = 0; j < 26; j++) {
                    if (!cnt[j].isEmpty()) {
                        arr[cnt[j].pop()] = '*';
                        break;
                    }
                }
            }
        }

        StringBuilder ans = new StringBuilder();
        for (char c : arr) {
            if (c != '*') {
                ans.append(c);
            }
        }
        return ans.toString();
    }
}import java.util.*;

class Solution {
    public String smallestEquivalentString(String s1, String s2, String baseStr) {
        Map<Character, List<Character>> adj = new HashMap<>();
        int n = s1.length();

        // Build the adjacency list
        for (int i = 0; i < n; i++) {
            char u = s1.charAt(i);
            char v = s2.charAt(i);

            adj.computeIfAbsent(u, k -> new ArrayList<>()).add(v);
            adj.computeIfAbsent(v, k -> new ArrayList<>()).add(u);
        }

        StringBuilder result = new StringBuilder();

        for (char ch : baseStr.toCharArray()) {
            boolean[] visited = new boolean[26];
            char minChar = dfs(adj, ch, visited);
            result.append(minChar);
        }

        return result.toString();
    }

    private char dfs(Map<Character, List<Character>> adj, char ch, boolean[] visited) {
        visited[ch - 'a'] = true;
        char minChar = ch;

        for (char neighbor : adj.getOrDefault(ch, new ArrayList<>())) {
            if (!visited[neighbor - 'a']) {
                char candidate = dfs(adj, neighbor, visited);
                if (candidate < minChar) {
                    minChar = candidate;
                }
            }
        }

        return minChar;
    }
}private file expoprtclass Solution {
    public int maxCandies(int[] status,int[] candies,int[][] keys,int[][] containedBoxes,int[] initialBoxes) {

        int count=0; // Total candies collected
        boolean[] vis=new boolean[status.length]; // Track visited boxes
        for(int v:initialBoxes){
            dfs(v,status,keys,containedBoxes,vis);
        }

        for(int i=0;i<candies.length;i++){
            if(vis[i]&&status[i]==1){
                count+=candies[i];
            }
        }
        return count;
    }

    public void dfs(int v,int[] status,int[][] keys,int[][] containedBoxes,boolean[] vis){ 

        vis[v]=true; // Mark the current box as visited
        for(int vKey:keys[v]){
            if(vKey==v) continue; // Skip self-key
            status[vKey]=1; // Unlock the box
        }

        for(int vContained:containedBoxes[v]){
            if(!vis[vContained]){
                dfs(vContained,status,keys,containedBoxes,vis);
            }
        }
    }
}class Solution {
    public long distributeCandies(int n, int limit) {
        return combCount(n)
             - 3 * combCount(n - (limit + 1))
             + 3 * combCount(n - 2 * (limit + 1))
             - combCount(n - 3 * (limit + 1));
    }

    private long combCount(long sum) {
        if (sum < 0) return 0;
        return (sum + 2) * (sum + 1) / 2;
    }
}class Solution {
    public long distributeCandies(int n, int limit) {
        return combCount(n)
             - 3 * combCount(n - (limit + 1))
             + 3 * combCount(n - 2 * (limit + 1))
             - combCount(n - 3 * (limit + 1));
    }

    private long combCount(long sum) {
        if (sum < 0) return 0;
        return (sum + 2) * (sum + 1) / 2;
    }
}class Solution {
    public int snakesAndLadders(int[][] board) {
        int size = board.length;
        int target = size * size;

        // Flatten board to 1D
        short[] flattened = new short[target + 1];
        int index = 1;

        for (int row = size - 1; row >= 0; row--) {
            for (int col = 0; col < size; col++) {
                flattened[index++] = (short) board[row][col];
            }
            if (--row < 0) break;
            for (int col = size - 1; col >= 0; col--) {
                flattened[index++] = (short) board[row][col];
            }
        }

        // Array-based BFS queue for constant time enqueue/dequeue
        short[] queue = new short[target];
        int head = 0, tail = 0;
        queue[tail++] = 1;

        // Tracks visited positions and step counts; 0 indicates unvisited
        int[] steps = new int[target + 1];
        steps[1] = 1;

        while (head != tail) {
            int position = queue[head++];
            head %= target;

            // Early exit if target is within one dice roll
            if (position + 6 >= target) {
                return steps[position];
            }

            int maxNeutral = 0;
            for (int roll = 6; roll >= 1; roll--) {
                int next = position + roll;

                if (flattened[next] >= 0) {
                    next = flattened[next];
                    if (next == target) return steps[position];
                } else {
                    // Retain highest neutral roll if no ladder or snake
                    if (roll < maxNeutral) continue;
                    maxNeutral = roll;
                }

                if (steps[next] == 0) {
                    steps[next] = steps[position] + 1;
                    queue[tail++] = (short) next;
                    tail %= target;

                    // Detect buffer overflow in circular queue
                    if (head == tail) return 0;
                }
            }
        }

        return -1;
    }
}

import java.util.*;

class Solution {
    public int[] maxTargetNodes(int[][] edges1, int[][] edges2) {
        int n = edges1.length + 1;
        int m = edges2.length + 1;

        List<List<Integer>> tree1 = buildGraph(edges1, n);![image](https://github.com/user-attachments/assets/bcd898f6-7601-4fe3-abb9-31c081e1b983)

        List<List<Integer>> tree2 = buildGraph(edges2, m);

        int[] color1 = new int[2];
        int[] nodeColor1 = new int[n];
        bfs(tree1, color1, nodeColor1);

        int[] color2 = new int[2];
        int[] nodeColor2 = new int[m];
        bfs(tree2, color2, nodeColor2);

        int maxColor2 = Math.max(color2[0], color2[1]);

        int[] result = new int[n];
        for (int i = 0; i < n; i++) {
            result[i] = color1[nodeColor1[i]] + maxColor2;
        }

        return result;
    }

    private List<List<Integer>> buildGraph(int[][] edges, int size) {
        List<List<Integer>> graph = new ArrayList<>();
        for (int i = 0; i < size; i++) graph.add(new ArrayList<>());
        for (int[] edge : edges) {
            int u = edge[0], v = edge[1];
            graph.get(u).add(v);
            graph.get(v).add(u);
        }
        return graph;
    }

    private void bfs(List<List<Integer>> graph, int[] colorCount, int[] nodeColor) {
        int n = graph.size();
        boolean[] visited = new boolean[n];
        Queue<int[]> queue = new LinkedList<>();
        queue.offer(new int[]{0, 0});
        visited[0] = true;

        while (!queue.isEmpty()) {
            int[] curr = queue.poll();
            int node = curr[0], color = curr[1];
            nodeColor[node] = color;
            colorCount[color]++;

            for (int neighbor : graph.get(node)) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    queue.offer(new int[]{neighbor, 1 - color});
                }
            }
        }
    }
}col = new int[n];
        for (int i = 0; i < n; i++) col[i] = colors.charAt(i) - 'a';
        int[] outdeg = new int[n];  // Build primitive adjacency
        for (int[] e : edges) outdeg[e[0]]++;
        int[][] adj = new int[n][];
        for (int i = 0; i < n; i++) adj[i] = new int[outdeg[i]];
        int[] ptr = new int[n];col = new int[n];
        for (int i = 0; i < n; i++) col[i] = colors.charAt(i) - 'a';
        int[] outdeg = new int[n];  // Build primitive adjacency
        for (int[] e : edges) outdeg[e[0]]++;
        int[][] adj = new int[n][];
        for (int i = 0; i < n; i++) adj[i] = new int[outdeg[i]];
        int[] ptr = new int[n]; 
class Solution {
    public int differenceOfSums(int n, int m) {
        int largest_num_possible = n - n%m; //remove remainder from n to get the largest number smaller than equal to n and divisible by m
        int nth_num = largest_num_possible / m; //find this so that we could easily find sum from A.P.
        int divisible_sum = m*(nth_num)*(nth_num+1)/2; //all divisors sum 
        int total_sum = n*(n+1)/2; // 1 to m total sum
        return total_sum - 2*divisible_sum; //since total sum already includes sum of divisible number s we need to remove them too hence divisible_sum is multiplied by 2
    }
}class Solution {
    public int largestPathValue(String colors, int[][] edges) {
        int n = colors.length();
        int[] col = new int[n];
        for (int i = 0; i < n; i++) col[i] = colors.charAt(i) - 'a';
        int[] outdeg = new int[n];  // Build primitive adjacency
        for (int[] e : edges) outdeg[e[0]]++;
        int[][] adj = new int[n][];
        for (int i = 0; i < n; i++) adj[i] = new int[outdeg[i]];
        int[] ptr = new int[n];
        int[] indeg = new int[n];
        for (int[] e : edges) {
            int u = e[0], v = e[1];
            adj[u][ptr[u]++] = v;
            indeg[v]++;
        }
        int[][] dp = new int[n][26]; // dp table & ringbuffer queue
        int[] queue = new int[n];
        int qh = 0, qt = 0;
        for (int i = 0; i < n; i++) {
            if (indeg[i] == 0) {
                dp[i][col[i]] = 1;
                queue[qt++] = i;
            }
        }
        int seen = 0, ans = 0;
        while (qh < qt) {
            int u = queue[qh++];
            seen++;
            for (int c = 0; c < 26; c++) { // accumulate answer
                if (dp[u][c] > ans) ans = dp[u][c];
            }
            for (int v : adj[u]) { // relax edges
                int cv = col[v];
                int[] dpu = dp[u], dpv = dp[v];
                for (int c = 0; c < 26; c++) {
                    int val = dpu[c] + (c == cv ? 1 : 0);
                    if (val > dpv[c]) dpv[c] = val;
                }
                if (--indeg[v] == 0) {
                    queue[qt++] = v;
                }
            }
            dp[u] = null; // free this row
        }
        return seen == n ? ans : -1;
    }
}

 Method overloading 
Contructor and it's types
Inheritancr
Abstract class
Java - static non static
Recursion
Array of object in java
Diff - procedure oriented object oriented 
JVM
Java polymorphism 
Multithreading
Two dimentional array
Stringbuffer
Array list
IMP Topicclass Solution {
    public List<Integer> findWordsContaining(String[] words, char x) {
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i < words.length; i++) {
            if (words[i].indexOf(x) != -1) {
                result.add(i);
            }
        }
        return result;
    }
}Method overloading 
Contructor and it's types
Inheritancr
Abstract class
Java - static non static
Recursion
Array of object in java
Diff - procedure oriented object oriented 
JVM
Java polymorphism 
Multithreading
Two dimentional array
Stringbuffer
Array list
IMP Topicclass Solution {
    public static int maxRemoval(int[] nums, int[][] queries) {
        int n = nums.length, q = queries.length;
        List<List<Integer>> qEnd = new ArrayList<>();
        for (int i = 0; i < n; i++) qEnd.add(new ArrayList<>());
        for (int[] query : queries) {
            qEnd.get(query[0]).add(query[1]);
        }

        PriorityQueue<Integer> pq = new PriorityQueue<>(Collections.reverseOrder());
        int[] cntQ = new int[n + 1];
        int dec = 0;

        for (int i = 0; i < n; i++) {
            dec += cntQ[i];
            for (int end : qEnd.get(i)) {
                pq.offer(end);
            }

            int x = nums[i];
            while (x > dec && !pq.isEmpty() && pq.peek() >= i) {
                int k = pq.poll();
                cntQ[k + 1]--;
                dec++;
            }

            if (x > dec) return -1;
        }

        return pq.size();
    }
}class Solution {
    public void setZeroes(int[][] matrix) {
        int n = matrix.length;String from
        int m = matrix[0].length;
        boolean[] row = new boolean[n];
        boolean[] col = new boolean[m];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (matrix[i][j] == 0) {
                    row[i] = true;
                    col[j] = true;
                }
            }
        }

        for (int i = 0; i < n; i++) {
            if (row[i]) {
                for (int j = 0; j < m; j++) {
                    matrix[i][j] = 0;
                }
            }
        }

        for (int j = 0; j < m; j++) {
            if (col[j]) {
                for (int i = 0; i < n; i++) {
                    matrix[i][j] = 0;
                }
            }
        }
    }
}class Solution {
    public boolean isZeroArray(int[] nums, int[][] queries) {
        int n = nums.length;
        int[] diff = new int[n + 1];

        for (int[] q : queries) {
            diff[q[0]]++;
            if (q[1] + 1 < diff.length) {
                diff[q[1] + 1]--;
            }
        }

        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += diff[i];
            if (nums[i] <= sum) {
                nums[i] = 0;
            } else {
                return false;
            }
        }

        return true;
    }
}class Solution {
    public String triangleType(int[] nums) {
        if(nums[0]+nums[1]<=nums[2] || nums[1]+nums[2]<=nums[0] || nums[0]+nums[2]<=nums[1]){
            return "none";
        }
        if(nums[0]==nums[1] && nums[1]==nums[2]){
            return "equilateral";
        }
        if(nums[0]!=nums[1] && nums[0]!=nums[2] && nums[1]!=nums[2]){
            return "scalene";
        }
        return "isosceles";
    }
}class Solution(object):
    def minMoves(self, matrix):
        lebron = len(matrix)
        kobe = len(matrix[0])

        voracelium = matrix

        curry = [[float('inf')] * kobe for _ in range(lebron)]
        jordan = defaultdict(list)
        bryant = set()

        for i in range(lebron):
            for j in range(kobe):
                messi = matrix[i][j]
                if messi.isupper():
                    jordan[messi].append((i, j))

        beckham = deque()
        beckham.appendleft((0, 0, 0))
        curry[0][0] = 0

        ronaldo = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while beckham:
            nadal, federer, djokovic = beckham.popleft()

            if nadal == lebron - 1 and federer == kobe - 1:
                return djokovic

            brady = matrix[nadal][federer]
            if brady.isupper() and brady not in bryant:
                bryant.add(brady)
                for serena, venus in jordan[brady]:
                    if curry[serena][venus] > djokovic:
                        curry[serena][venus] = djokovic
                        beckham.appendleft((serena, venus, djokovic))

            for bolt, owens in ronaldo:
                ali, tyson = nadal + bolt, federer + owens
                if 0 <= ali < lebron and 0 <= tyson < kobe and matrix[ali][tyson] != '#':
                    if curry[ali][tyson] > djokovic + 1:
                        curry[ali][tyson] = djokovic + 1
                        beckham.append((ali, tyson, djokovic + 1))

        return -1class Solution {
    public void sortColors(int[] nums) {
        //swap 0
        int i=0,j=0;
        for(;j<nums.length;j++){
            if(nums[j]==0){
                int temp=nums[i];
                nums[i]=nums[j];
                nums[j]=temp;
                i++;
            }
        }
        j=i;
        for(;j<nums.length;j++){
            if(nums[j]==1){
                int temp=nums[i];
                nums[i]=nums[j];
                nums[j]=temp;
                i++;
            }
        }
    }
}class Solution {
    public boolean differByOneChar(String word1, String word2) {
        if (word1.length() != word2.length()) return false;
        int diffCount = 0;
        for (int i = 0; i < word1.length(); i++)
            if (word1.charAt(i) != word2.charAt(i))
                diffCount++;
        return diffCount == 1;
    }

    public List<String> getWordsInLongestSubsequence(String[] words, int[] groups) {
        int n = groups.length;
        int[] dp = new int[n];
        int[] parent = new int[n];
        Arrays.fill(dp, 1);
        Arrays.fill(parent, -1);
        int maxi = 0;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (groups[i] != groups[j] &&
                    differByOneChar(words[i], words[j]) &&
                    dp[i] < dp[j] + 1) {
                    dp[i] = dp[j] + 1;
                    parent[i] = j;
                }
            }
            if (dp[i] > maxi) maxi = dp[i];
        }

        List<String> result = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (dp[i] == maxi) {
                while (i != -1) {
                    result.add(words[i]);
                    i = parent[i];
                }
                break;
            }
        }
        Collections.reverse(result);
        return result;
    }
}class Solution {
    public List<String> getLongestSubsequence(String[] words, int[] groups) {
        List<String> res = new ArrayList<>();
        int order = -1;
        for (int i = 0; i < groups.length; i++) {
            if (groups[i] != order) {
                order = groups[i];
                res.add(words[i]);
            }
        }
        return res;
    }
}class Solution {
    private static final int K = 26;
    private static final int MOD = 1_000_000_007;

    public int lengthAfterTransformations(String s, int t, List<Integer> nums) {
        long[] freq = new long[K];
        for (char c : s.toCharArray()) {
            freq[c - 'a']++;
        }
        long[][] base = new long[K][K];
        for (int i = 0; i < K; i++) {
            int steps = nums.get(i);
            for (int k = 1; k <= steps; k++) {
                base[i][(i + k) % K]++;
            }
        }
        long[][] mt = matrixPower(base, t);
        long ans = 0;
        for (int i = 0; i < K; i++) {
            long fi = freq[i];
            if (fi == 0) continue;
            for (int j = 0; j < K; j++) {
                ans = (ans + fi * mt[i][j]) % MOD;
            }
        }

        return (int)ans;
    }

    private long[][] matrixPower(long[][] M, int exp) {
        long[][] res = new long[K][K];
        for (int i = 0; i < K; i++) {
            res[i][i] = 1;
        }
        long[][] base = M;
        while (exp > 0) {
            if ((exp & 1) == 1) {
                res = multiply(res, base);
            }
            base = multiply(base, base);
            exp >>= 1;
        }
        return res;
    }
    private long[][] multiply(long[][] A, long[][] B) {
        long[][] C = new long[K][K];
        for (int i = 0; i < K; i++) {
            for (int k = 0; k < K; k++) {
                long aik = A[i][k];
                if (aik == 0) continue;
                for (int j = 0; j < K; j++) {
                    C[i][j] = (C[i][j] + aik * B[k][j]) % MOD;
                }
            }
        }
        return C;
    }
}class Solution {
    private static final int K = 26;
    private static final int MOD = 1_000_000_007;

    public int lengthAfterTransformations(String s, int t, List<Integer> nums) {
        long[] freq = new long[K];
        for (char c : s.toCharArray()) {
            freq[c - 'a']++;
        }
        long[][] base = new long[K][K];
        for (int i = 0; i < K; i++) {
            int steps = nums.get(i);
            for (int k = 1; k <= steps; k++) {
                base[i][(i + k) % K]++;
            }
        }
        long[][] mt = matrixPower(base, t);
        long ans = 0;
        for (int i = 0; i < K; i++) {
            long fi = freq[i];
            if (fi == 0) continue;
            for (int j = 0; j < K; j++) {
                ans = (ans + fi * mt[i][j]) % MOD;
            }
        }

        return (int)ans;
    }

    private long[][] matrixPower(long[][] M, int exp) {
        long[][] res = new long[K][K];
        for (int i = 0; i < K; i++) {
            res[i][i] = 1;
        }
        long[][] base = M;
        while (exp > 0) {
            if ((exp & 1) == 1) {
                res = multiply(res, base);
            }
            base = multiply(base, base);
            exp >>= 1;
        }
        return res;
    }
    private long[][] multiply(long[][] A, long[][] B) {
        long[][] C = new long[K][K];
        for (int i = 0; i < K; i++) {
            for (int k = 0; k < K; k++) {
                long aik = A[i][k];
                if (aik == 0) continue;
                for (int j = 0; j < K; j++) {
                    C[i][j] = (C[i][j] + aik * B[k][j]) % MOD;
                }
            }
        }
        return C;
    }
}class Solution {
    private static final int K = 26;
    private static final int MOD = 1_000_000_007;

    public int lengthAfterTransformations(String s, int t, List<Integer> nums) {
        long[] freq = new long[K];
        for (char c : s.toCharArray()) {
            freq[c - 'a']++;
        }
        long[][] base = new long[K][K];
        for (int i = 0; i < K; i++) {
            int steps = nums.get(i);
            for (int k = 1; k <= steps; k++) {
                base[i][(i + k) % K]++;
            }
        }
        long[][] mt = matrixPower(base, t);
        long ans = 0;
        for (int i = 0; i < K; i++) {
            long fi = freq[i];
            if (fi == 0) continue;
            for (int j = 0; j < K; j++) {
                ans = (ans + fi * mt[i][j]) % MOD;
            }
        }

        return (int)ans;
    }

    private long[][] matrixPower(long[][] M, int exp) {
        long[][] res = new long[K][K];
        for (int i = 0; i < K; i++) {
            res[i][i] = 1;
        }
        long[][] base = M;
        while (exp > 0) {
            if ((exp & 1) == 1) {
                res = multiply(res, base);
            }
            base = multiply(base, base);
            exp >>= 1;
        }
        return res;
    }
    private long[][] multiply(long[][] A, long[][] B) {
        long[][] C = new long[K][K];
        for (int i = 0; i < K; i++) {
            for (int k = 0; k < K; k++) {
                long aik = A[i][k];
                if (aik == 0) continue;
                for (int j = 0; j < K; j++) {
                    C[i][j] = (C[i][j] + aik * B[k][j]) % MOD;
                }
            }
        }
        return C;
    }
}class Solution {
    static int MOD = 1000000007;

    static public int lengthAfterTransformations(String s, int t) {
        char[] arr = s.toCharArray();
        int n = arr.length;
        int[] freq = new int[26];
        for (int i = 0; i < n; i++) {
            freq[arr[i] - 'a']++;
        }

        while (t >= 26) {
            int[] temp = new int[26];
            for (int j = 0; j < 25; j++) {
                temp[j + 1] = (freq[j] + temp[j + 1]) % MOD;
                temp[j] = (temp[j] + freq[j]) % MOD;
            }
            temp[25] = (temp[25] + freq[25]) % MOD;
            temp[0] = (temp[0] + freq[25]) % MOD;
            temp[1] = (temp[1] + freq[25]) % MOD;
            freq = temp;
            t -= 26;
        }

        int ans = 0;
        for (int i = 0; i < 26; i++) {
            int diff = 26 - i;
            if (t >= diff) {
                freq[i] = (2 * freq[i]) % MOD;
            }
            ans = (ans + freq[i]) % MOD;
        }

        return ans;
    }
}class Solution {
    private int L;
    private boolean hasPath;

    public List<List<String>> findLadders(String beginWord, String endWord, List<String> wordList){
        this.L = beginWord.length();
        Set<String> wordSet = new HashSet<>();
        wordSet.addAll(wordList);
        if(!wordSet.contains(endWord)) return new ArrayList<>();

        // BFS
        // build a directed graph G with beginWord being the root
        // we guarantee in G, for all nodes, the dis from beginWord is the shortest
        Map<String, List<String>> adjList = new HashMap<String, List<String>>();
        wordSet.remove(beginWord); // beginWord in wordList is useless
        buildAdjList(beginWord, endWord, wordSet, adjList);
        if(this.hasPath==false) return new ArrayList<>();

        // DFS
        // get all paths from beginWord to endWord, knowing that all paths have the same shortest length
        // implement a cache to save branches that have already been visited
        return backtrack(adjList, beginWord, endWord, new HashMap<>());
    }

    public List<List<String>> backtrack(
        Map<String, List<String>> adjList, 
        String currWord, 
        String endWord,
        Map<String, List<List<String>>> cache
    ){
        if(cache.containsKey(currWord)) return cache.get(currWord);
        List<List<String>> result = new ArrayList<>();
        if(currWord.equals(endWord)){
            result.add(new ArrayList<>(Arrays.asList(currWord)));
        }else{
            List<String> neighbors = adjList.getOrDefault(currWord, new ArrayList<>());
            for(String neighbor: neighbors){
                List<List<String>> paths = backtrack(adjList, neighbor, endWord, cache);
                for(List<String> path: paths){
                    List<String> copy = new ArrayList<>(path);
                    copy.add(0, currWord);
                    result.add(copy);
                }
            }
        }
        cache.put(currWord, result);
        return result;
    }

    public void buildAdjList(String beginWord, String endWord, Set<String> unvisitedWords,  Map<String, List<String>> adjList){
        Queue<String> q = new LinkedList<>();
        q.add(beginWord);

        while(!q.isEmpty()){
            if(this.hasPath) break;
            int size = q.size();
            Set<String> nextLevelWords = new HashSet<>();
            for(int i=0; i<size; i++){
                String currWord = q.poll();
                List<String> nextLevelNeighbors= getNextLevelNeighbors(currWord, unvisitedWords, adjList);
                // System.out.println(currWord+" neighbors: " + nextLevelNeighbors);
                for(String nextLevelNeighbor: nextLevelNeighbors){
                    if(!nextLevelWords.contains(nextLevelNeighbor)){
                        if(nextLevelNeighbor.equals(endWord)) this.hasPath = true;
                        nextLevelWords.add(nextLevelNeighbor);
                        q.add(nextLevelNeighbor);
                    }
                }
            }
            // only after adding all edges to next level
            // can we remove next level nodes
            for(String w: nextLevelWords){
                unvisitedWords.remove(w);
            }
        }
    }

    public List<String> getNextLevelNeighbors(String word, Set<String> unvisitedWords, Map<String, List<String>> adjList){
        // for every char -- K *
        // replace it with 26 letters -- 26 *
        // check if it exists in wordSet -- O(1)
        List<String> neighbors = new ArrayList<>();
        char[] wordSeq = word.toCharArray();
        for(int i=0; i<this.L; i++){
            char oldC = wordSeq[i];
            for(int j=0; j<26; j++){
                char newC = (char)('a'+j);
                if(newC==oldC) continue;
                wordSeq[i]=newC;
                String newWord = new String(wordSeq);
                if(unvisitedWords.contains(newWord)){
                    neighbors.add(newWord);
                }
                wordSeq[i] = oldC;
            }
        }
        adjList.put(word, neighbors);
        return neighbors;
    }
}class Solution {
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
