package algorithm;

import data.State;

import java.util.*;

public class VariousAlgorithm {

    /**
     * created by Sven
     * on 2019-12-10
     * <p>
     * 爬楼梯
     * <p>
     * 假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
     * 每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
     * 注意：给定 n 是一个正整数。
     */

    public int climbStairs(int n) {
        int[] climbStepCount = new int[n];
        if (n == 1)
            return 1;
        if (n == 2)
            return 2;
        climbStepCount[0] = 1;
        climbStepCount[1] = 2;
        for (int i = 2; i < n; i++) {
            climbStepCount[i] = climbStepCount[i - 2] + climbStepCount[i - 1];
        }
        return climbStepCount[n - 1];
    }


    /**
     * created by Sven
     * on 2019-12-03
     * <p>
     * X的平方根
     * <p>
     * 实现 int sqrt(int x) 函数。
     * <p>
     * 计算并返回 x 的平方根，其中 x 是非负整数。
     * <p>
     * 由于返回类型是整数，结果只保留整数的部分，小数部分将被舍去。
     */

    public int mySqrt(int x) {
        if (x < 0)
            return 0;
        if (x < 2)
            return x;
        for (long i = 1; i <= x / 2; i++) {
            long l = (i + 1) * (i + 1);
            if (l > Integer.MAX_VALUE || l > x)
                return (int) i;
        }
        return 0;
    }

    /**
     * 牛顿法
     *
     * @param x
     * @return
     */
    public int mySqrt2(int x) {
        if (x < 0)
            return 0;
        long left = 0;
        long right = x / 2 + 1;
        while (left < right) {
            long mid = (left + right + 1) / 2;
            long square = mid * mid;
            if (square > x) {
                right = mid - 1;
            } else if (square < x) {
                left = mid;
            } else {
                return (int) mid;
            }
        }
        return (int) left;
    }

    /**
     * 给你一个整数 n，请你帮忙计算并返回该整数「各位数字之积」与「各位数字之和」的差。
     */
    public int subtractProductAndSum(int n) {
        if (n == 0)
            return 0;
        int count = 0;
        int product = 1;
        while (n / 10 > 0) {
            count += (n % 10);
            product *= (n % 10);
            n = n / 10;
        }
        count += n;
        product *= n;
        return product - count;
    }

    /**
     * 买卖股票的最佳时机
     * <p>
     * 给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
     * <p>
     * 如果你最多只允许完成一笔交易（即买入和卖出一支股票），设计一个算法来计算你所能获取的最大利润。
     * <p>
     * 注意你不能在买入股票前卖出股票。
     *
     * @param prices
     * @return
     */
    public int maxProfit(int[] prices) {
        int minPrice = Integer.MAX_VALUE;
        int spreadPrice = 0;
        if (prices == null || prices.length == 0)
            return 0;
        for (int i = 0; i < prices.length; i++) {
            if (prices[i] < minPrice)
                minPrice = prices[i];
            else if (prices[i] - minPrice > spreadPrice) {
                spreadPrice = prices[i] - minPrice;
            }
        }
        return spreadPrice;
    }

    /**
     * 买卖股票的最佳时机 II
     * <p>
     * 给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
     * <p>
     * 设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。
     * <p>
     * 注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
     *
     * @param prices
     * @return
     */
    public int maxProfit2(int[] prices) {
        boolean isBuy = false;
        int buyPrice = 0;
        int spreadPriceCount = 0;
        if (prices == null || prices.length == 0)
            return 0;
        for (int i = 0; i < prices.length - 1; i++) {
            if (isBuy) {
                if (prices[i] > prices[i + 1]) {
                    spreadPriceCount = spreadPriceCount + prices[i] - buyPrice;
                    isBuy = false;
                }
            } else {
                if (prices[i] < prices[i + 1]) {
                    buyPrice = prices[i];
                    isBuy = true;
                }
            }
        }
        if (isBuy) {
            spreadPriceCount = spreadPriceCount + prices[prices.length - 1] - buyPrice;
        }
        return spreadPriceCount;
    }

    /**
     * Excel表列名称
     * <p>
     * 给定一个正整数，返回它在 Excel 表中相对应的列名称。
     *
     * @param n
     * @return
     */
    public String convertToTitle(int n) {
        if (n <= 0)
            return null;
        StringBuffer resultStringBuffer = new StringBuffer();
        while (n > 0) {
            int remainder = n % 26;
            if (remainder == 0) {
                remainder = 26;
                n--;
            }
            resultStringBuffer.insert(0, (char) ('A' + remainder - 1));
            n = n / 26;
        }
        return resultStringBuffer.toString();
    }

    /**
     * Excel表列序号
     * <p>
     * 给定一个Excel表格中的列名称，返回其相应的列序号。
     *
     * @param s
     * @return
     */
    public int titleToNumber(String s) {
        if (s == null || s.length() == 0)
            return 0;
        int result = 0;
        for (int i = 0; i < s.length(); i++) {
            int number = (int) (s.charAt(i) - 'A' + 1);
            result = result * 26 + number;
        }
        return result;
    }

    /**
     * 阶乘后的零
     * <p>
     * 给定一个整数 n，返回 n! 结果尾数中零的数量。
     *
     * @param n
     * @return
     */
    public int trailingZeroes(int n) {
        if (n < 5)
            return 0;
        long i = 5;
        int count = 0;
        while (i <= n) {
            long N = i;
            while (N > 0) {
                if (N % 5 == 0) {
                    count++;
                    N /= 5;
                } else {
                    break;
                }
            }
            i += 5;
        }

        return count;
    }

    /**
     * 颠倒二进制位
     * <p>
     * 颠倒给定的 32 位无符号整数的二进制位。
     *
     * @param n
     * @return
     */
    public int reverseBits(int n) {
        int res = 0;
        int count = 0;
        while (count < 32) {
            res <<= 1;
            res |= (n & 1);
            n >>= 1;
            count++;
        }
        return res;
    }

    /**
     * 位1的个数
     * <p>
     * 编写一个函数，输入是一个无符号整数，返回其二进制表达式中数字位数为 ‘1’ 的个数（也被称为汉明重量）。
     *
     * @param n
     * @return
     */
    public int hammingWeight(int n) {
        int count = 0;
        int mask = 1;
        for (int i = 0; i < 32; i++) {
            if ((n & mask) != 0) {
                count++;
            }
            mask <<= 1;
        }
        return count;
    }

    /**
     * 快乐数
     * <p>
     * 编写一个算法来判断一个数是不是“快乐数”。
     * <p>
     * 一个“快乐数”定义为：对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和，然后重复这个过程直到这个数变为 1，也可能是无限循环但始终变不到 1。如果可以变为 1，那么这个数就是快乐数。
     *
     * @param n
     * @return
     */
    public boolean isHappy(int n) {
        int slow = n, fast = n;
        do {
            slow = bitSquareSum(slow);
            fast = bitSquareSum(fast);
            fast = bitSquareSum(fast);
        } while (slow != fast);
        return slow == 1;
    }

    private int bitSquareSum(int n) {
        int sum = 0;
        while (n > 0) {
            int square = n % 10;
            sum += (square * square);
            n /= 10;
        }
        return sum;
    }

    /**
     * 计数质数
     * <p>
     * 统计所有小于非负整数 n 的质数的数量。
     *
     * @param n
     * @return
     */
    public int countPrimes(int n) {
        if (n < 2)
            return 0;
        boolean[] isPrime = new boolean[n];
        Arrays.fill(isPrime, true);
        for (int i = 2; i * i <= n; i++) {
            for (int j = 2 * i; j < n; j += i)
                isPrime[j] = false;
        }
        int count = 0;
        for (int i = 2; i < n; i++) {
            if (isPrime[i])
                count++;
        }
        return count;
    }

    /**
     * 365. 水壶问题
     * 有两个容量分别为 x升 和 y升 的水壶以及无限多的水。请判断能否通过使用这两个水壶，从而可以得到恰好 z升 的水？
     * <p>
     * 如果可以，最后请用以上水壶中的一或两个来盛放取得的 z升 水。
     * <p>
     * 你允许：
     * <p>
     * 装满任意一个水壶
     * 清空任意一个水壶
     * 从一个水壶向另外一个水壶倒水，直到装满或者倒空
     *
     * @param x
     * @param y
     * @param z
     * @return
     */
    public boolean canMeasureWater(int x, int y, int z) {
        if (z > x + y)
            return false;
        if (z == 0 || z == x || z == y || z == x + y)
            return true;
        State initState = new State(0, 0);

        //广度优先遍历使用队列
        Queue<State> queue = new LinkedList<>();
        Set<State> visited = new HashSet<>();

        queue.offer(initState);
        visited.add(initState);

        while (!queue.isEmpty()) {
            State head = queue.poll();

            int curX = head.getX();
            int curY = head.getY();

            if (curX == z || curY == z || curX + curY == z) {
                return true;
            }
            // 从当前状态获得所有可能的下一步的状态
            List<State> nextSates = getNextState(curX, curY, x, y);
            for (State nextState : nextSates) {
                if (!visited.contains(nextState)) {
                    queue.offer(nextState);
                    // 添加到队列以后，必须马上设置为已经访问，否则会出现死循环
                    visited.add(nextState);
                }
            }
        }
        return false;
    }

    private List<State> getNextState(int curX, int curY, int x, int y) {
        List<State> nextStates = new ArrayList<>(8);

        // 以下两个状态，对应操作 1
        // 外部加水，使得 A 满
        State nextState1 = new State(x, curY);
        // 外部加水，使得 B 满
        State nextState2 = new State(curX, y);

        // 以下两个状态，对应操作 2
        // 把 A 清空
        State nextState3 = new State(0, curY);
        // 把 B 清空
        State nextState4 = new State(curX, 0);

        // 以下四个状态，对应操作 3
        // 从 A 到 B，使得 B 满，A 还有剩
        State nextState5 = new State(curX - (y - curY), y);
        // 从 A 到 B，此时 A 的水太少，A 倒尽，B 没有满
        State nextState6 = new State(0, curX + curY);

        // 从 B 到 A，使得 A 满，B 还有剩余
        State nextState7 = new State(x, curY - (x - curX));
        // 从 B 到 A，此时 B 的水太少，B 倒尽，A 没有满
        State nextState8 = new State(curX + curY, 0);

        // 没有满的时候，才需要加水
        if (curX < x)
            nextStates.add(nextState1);

        if (curY < y)
            nextStates.add(nextState2);


        // 有水的时候，才需要倒掉
        if (curX > 0)
            nextStates.add(nextState3);

        if (curY > 0)
            nextStates.add(nextState4);


        // 有剩余才倒
        if (curX - (y - curY) > 0)
            nextStates.add(nextState5);

        if (curY - (x - curX) > 0)
            nextStates.add(nextState7);


        // 倒过去倒不满才倒
        if (curX + curY < y)
            nextStates.add(nextState6);

        if (curX + curY < x)
            nextStates.add(nextState8);

        return nextStates;
    }

    /**
     * 面试题 17.16. 按摩师
     * <p>
     * 一个有名的按摩师会收到源源不断的预约请求，每个预约都可以选择接或不接。
     * 在每次预约服务之间要有休息时间，因此她不能接受相邻的预约。给定一个预约请求序列，
     * 替按摩师找到最优的预约集合（总预约时间最长），返回总的分钟数。
     * <p>
     * 动态规划
     *
     * @param nums
     * @return
     */
    public int massage(int[] nums) {
        if (nums == null || nums.length == 0)
            return 0;
        if (nums.length == 1)
            return nums[0];

        if (nums.length == 2)
            return Math.max(nums[0], nums[1]);
        int[] result = new int[nums.length];
        result[0] = nums[0];
        result[1] = Math.max(nums[0], nums[1]);
        for (int i = 2; i < nums.length; i++) {
            result[i] = Math.max(result[i - 1], result[i - 2] + nums[i]);
        }
        return result[nums.length - 1];
    }

    /**
     * 892. 三维形体的表面积
     * <p>
     * 在 N * N 的网格上，我们放置一些 1 * 1 * 1  的立方体。
     * <p>
     * 每个值 v = grid[i][j] 表示 v 个正方体叠放在对应单元格 (i, j) 上。
     * <p>
     * 请你返回最终形体的表面积。
     *
     * @param grid
     * @return
     */
    public int surfaceArea(int[][] grid) {
        int[] dr = new int[]{0, 1, 0, -1};
        int[] dc = new int[]{1, 0, -1, 0};

        int length = grid.length;
        int result = 0;
        for (int r = 0; r < length; r++) {
            for (int c = 0; c < length; c++) {
                if (grid[r][c] > 0) {
                    result += 2;
                    for (int k = 0; k < 4; k++) {
                        int nr = r + dr[k];
                        int nc = c + dc[k];
                        int nv = 0;

                        if (0 <= nr && nr < length && 0 <= nc && nc < length)
                            nv = grid[nr][nc];
                        result += Math.max(grid[r][c] - nv, 0);
                    }
                }
            }
        }
        return result;
    }

    /**
     * 999. 车的可用捕获量
     *
     * @param board
     * @return
     */
    public int numRookCaptures(char[][] board) {
        if (board == null || board.length == 0)
            return 0;
        // 先找到R
        int i = 0, j = 0, result = 0;
        boolean isFind = false;
        while (i < board.length) {
            while (j < board[i].length) {
                if (board[i][j] == 'R') {
                    isFind = true;
                    break;
                }
                j++;
            }
            if (isFind) {
                break;
            }
            j = 0;
            i++;
        }
        int k = j + 1;
        while (k < board[i].length) {

            if (board[i][k] == '.') {

            } else if (board[i][k] == 'B')
                break;
            else if (board[i][k] == 'p') {
                result++;
                break;
            }
            k++;
        }
        k = j - 1;
        while (k >= 0) {
            if (board[i][k] == '.') {

            } else if (board[i][k] == 'B')
                break;
            if (board[i][k] == 'p') {
                result++;
                break;
            }
            k--;
        }
        k = i + 1;
        while (k < board.length) {
            if (board[k][j] == '.') {

            } else if (board[k][j] == 'B')
                break;
            if (board[k][j] == 'p') {
                result++;
                break;
            }
            k++;
        }
        k = i - 1;
        while (k >= 0) {
            if (board[k][j] == '.') {

            } else if (board[k][j] == 'B')
                break;
            if (board[k][j] == 'p') {
                result++;
                break;
            }
            k--;
        }
        return result;
    }

    /**
     * 1342. 将数字变成 0 的操作次数
     * <p>
     * 给你一个非负整数 num ，请你返回将它变成 0 所需要的步数。 如果当前数字是偶数，你需要把它除以 2 ；否则，减去 1 。
     *
     * @param num
     * @return
     */
    public int numberOfSteps(int num) {
        int result = 0;
        while (num > 0) {
            if (num % 2 == 0) {
                num = num >> 1;
                result++;
            } else {
                num--;
                if (num == 0) {
                    result++;
                    return result;
                } else {
                    num = num >> 1;
                    result += 2;
                }
            }

        }
        return result;
    }

    /**
     * 面试题62. 圆圈中最后剩下的数字
     * 0,1,,n-1这n个数字排成一个圆圈，从数字0开始，每次从这个圆圈里删除第m个数字。求出这个圆圈里剩下的最后一个数字。
     * <p>
     * 例如，0、1、2、3、4这5个数字组成一个圆圈，从数字0开始每次删除第3个数字，则删除的前4个数字依次是2、0、4、1，因此最后剩下的数字是3。
     *
     * @param n
     * @param m
     * @return
     */
    public int lastRemaining(int n, int m) {
        if (m == 0)
            return 0;
        ArrayList<Integer> list = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            list.add(i);
        }
        int idx = 0;
        while (n > 1) {
            idx = (idx + m - 1) % n;
            list.remove(idx);
            n--;
        }
        return list.get(0);
    }

    public int lastRemaining_2(int n, int m) {
        int result = 0;
        for (int i = 2; i <= n; i++) {
            result = (result + m) % i;
        }
        return result;
    }

    /**
     * 面试题13. 机器人的运动范围
     * 地上有一个m行n列的方格，从坐标 [0,0] 到坐标 [m-1,n-1] 。一个机器人从坐标 [0, 0] 的格子开始移动，
     * 它每次可以向左、右、上、下移动一格（不能移动到方格外），也不能进入行坐标和列坐标的数位之和大于k的格子。
     * 例如，当k为18时，机器人能够进入方格 [35, 37] ，因为3+5+3+7=18。但它不能进入方格 [35, 38]，因为3+5+3+8=19。
     * 请问该机器人能够到达多少个格子？
     *
     * @param m
     * @param n
     * @param k
     * @return
     */
    public int movingCount(int m, int n, int k) {
        if (m < 0 || n < 0)
            return 0;
        boolean[][] visited = new boolean[m][n];
        int res = 0;
        Queue<int[]> queue = new LinkedList<int[]>();
        queue.add(new int[]{0, 0, 0, 0});
        while (queue.size() > 0) {
            int[] x = queue.poll();
            int i = x[0], j = x[1], si = x[2], sj = x[3];
            if (i >= m || j >= n || k < sj + si || visited[i][j])
                continue;
            visited[i][j] = true;
            res++;

            queue.add(new int[]{i + 1, j, (i + 1) % 10 != 0 ? si + 1 : si - 8, sj});
            queue.add(new int[]{i, j + 1, si, (j + 1) % 10 != 0 ? sj + 1 : sj - 8});
        }
        return res;

    }

    /**
     * 22. 括号生成
     * 给出 n 代表生成括号的对数，请你写出一个函数，使其能够生成所有可能的并且有效的括号组合。
     * <p>
     * 例如，给出 n = 3，生成结果为：
     *
     * @param n [
     *          "((()))",
     *          "(()())",
     *          "(())()",
     *          "()(())",
     *          "()()()"
     *          ]
     * @return
     */
    public List<String> generateParenthesis(int n) {
        List<String> result = new ArrayList<>();
        backtrack(result, new StringBuilder(), 0, 0, n);
        return result;
    }

    private void backtrack(List<String> result, StringBuilder cur, int open, int close, int max) {
        if (cur.length() == max * 2) {
            result.add(cur.toString());
            return;
        }
        if (open < max) {
            cur.append('(');
            backtrack(result, cur, open + 1, close, max);
            cur.deleteCharAt(cur.length() - 1);
        }
        if (close < open) {
            cur.append(')');
            backtrack(result, cur, open, close + 1, max);
            cur.deleteCharAt(cur.length() - 1);
        }
    }

    /**
     * 887. 鸡蛋掉落
     * 你将获得 K 个鸡蛋，并可以使用一栋从 1 到 N  共有 N 层楼的建筑。
     * <p>
     * 每个蛋的功能都是一样的，如果一个蛋碎了，你就不能再把它掉下去。
     * <p>
     * 你知道存在楼层 F ，满足 0 <= F <= N 任何从高于 F 的楼层落下的鸡蛋都会碎，从 F 楼层或比它低的楼层落下的鸡蛋都不会破。
     * <p>
     * 每次移动，你可以取一个鸡蛋（如果你有完整的鸡蛋）并把它从任一楼层 X 扔下（满足 1 <= X <= N）。
     * <p>
     * 你的目标是确切地知道 F 的值是多少。
     * <p>
     * 无论 F 的初始值如何，你确定 F 的值的最小移动次数是多少？
     *
     * @param K
     * @param N
     * @return
     */
    public int superEggDrop(int K, int N) {
        return superEggDropDp(K, N);
    }

    Map<Integer, Integer> superMemo = new HashMap<>();

    private int superEggDropDp(int K, int N) {
        if (!superMemo.containsKey(N * 100 + K)) {
            int ans;
            if (N == 0)
                ans = 0;
            else if (K == 1)
                ans = N;
            else {
                int lo = 1, hi = N;
                while (lo + 1 < hi) {
                    int x = (lo + hi) / 2;
                    int t1 = superEggDrop(K - 1, x - 1);
                    int t2 = superEggDrop(K, N - x);
                    if (t1 < t2)
                        lo = x;
                    else if (t1 > t2)
                        hi = x;
                    else
                        lo = hi = x;
                }
                ans = 1 + Math.min(Math.max(superEggDropDp(K - 1, lo - 1), superEggDropDp(K, N - lo)), Math.max(superEggDropDp(K - 1, hi - 1), superEggDropDp(K, N - hi)));
            }
            superMemo.put(N * 100 + K, ans);
        }
        return superMemo.get(N * 100 + K);
    }


    /**
     * 面试题 16.03. 交点
     * <p>
     * 给定两条线段（表示为起点start = {X1, Y1}和终点end = {X2, Y2}），如果它们有交点，请计算其交点，没有交点则返回空值。
     * <p>
     * 要求浮点型误差不超过10^-6。若有多个交点（线段重叠）则返回 X 值最小的点，X 坐标相同则返回 Y 值最小的点。
     *
     * @param start1
     * @param end1
     * @param start2
     * @param end2
     * @return
     */
    public double[] intersection(int[] start1, int[] end1, int[] start2, int[] end2) {
        int x1 = start1[0], y1 = start1[1];
        int x2 = end1[0], y2 = end1[1];
        int x3 = start2[0], y3 = start2[1];
        int x4 = end2[0], y4 = end2[1];

        double[] ans = new double[2];
        Arrays.fill(ans, Double.MAX_VALUE);
        // 判断两直线是否平行
        if ((y4-y3)*(x2-x1) == (y2-y1)*(x4-x3)) {
            // 判断两直线是否重叠
            if ((y2-y1)*(x3-x1) == (y3-y1)*(x2-x1)) {
                // 判断 (x3, y3) 是否在「线段」(x1, y1)~(x2, y2) 上
                if (isInside(x1, y1, x2, y2, x3, y3)) {
                    updateRes(ans, x3, y3);
                }
                // 判断 (x4, y4) 是否在「线段」(x1, y1)~(x2, y2) 上
                if (isInside(x1, y1, x2, y2, x4, y4)) {
                    updateRes(ans, (double)x4, (double)y4);
                }
                // 判断 (x1, y1) 是否在「线段」(x3, y3)~(x4, y4) 上
                if (isInside(x3, y3, x4, y4, x1, y1)) {
                    updateRes(ans, (double)x1, (double)y1);
                }
                // 判断 (x2, y2) 是否在「线段」(x3, y3)~(x4, y4) 上
                if (isInside(x3, y3, x4, y4, x2, y2)) {
                    updateRes(ans, (double)x2, (double)y2);
                }
            }
        } else {
            // 联立方程得到 t1 和 t2 的值
            double t1 = (double)(x3 * (y4 - y3) + y1 * (x4 - x3) - y3 * (x4 - x3) - x1 * (y4 - y3)) / ((x2 - x1) * (y4 - y3) - (x4 - x3) * (y2 - y1));
            double t2 = (double)(x1 * (y2 - y1) + y3 * (x2 - x1) - y1 * (x2 - x1) - x3 * (y2 - y1)) / ((x4 - x3) * (y2 - y1) - (x2 - x1) * (y4 - y3));
            // 判断 t1 和 t2 是否均在 [0, 1] 之间
            if (t1 >= 0.0 && t1 <= 1.0 && t2 >= 0.0 && t2 <= 1.0) {
                ans[0] = x1 + t1 * (x2 - x1);
                ans[1] = y1 + t1 * (y2 - y1);
            }
        }
        if (ans[0] == Double.MAX_VALUE) {
            return new double[0];
        }
        return ans;
    }

    // 判断 (x, y) 是否在「线段」(x1, y1)~(x2, y2) 上
    // 这里的前提是 (x, y) 一定在「直线」(x1, y1)~(x2, y2) 上
    private boolean isInside(int x1, int y1, int x2, int y2, int x, int y) {
        // 若与 x 轴平行，只需要判断 x 的部分
        // 若与 y 轴平行，只需要判断 y 的部分
        // 若为普通线段，则都要判断
        return (x1 == x2 || (Math.min(x1, x2) <= x && x <= Math.max(x1, x2)))
                && (y1 == y2 || (Math.min(y1, y2) <= y && y <= Math.max(y1, y2)));
    }

    private void updateRes(double[] ans, double x, double y) {
        if (x < ans[0] || (x == ans[0] && y < ans[1])) {
            ans[0] = x;
            ans[1] = y;
        }
    }

    /**
     * 面试题 08.11. 硬币
     * 硬币。给定数量不限的硬币，币值为25分、10分、5分和1分，编写代码计算n分有几种表示法。(结果可能会很大，你需要将结果模上1000000007)
     * @param n
     * @return
     */
    public int waysToChange(int n) {
        if(n <= 0)
            return 0;
        if(n < 5)
            return 1;
        int mod = 1000000007;
        int result = 0;
        for(int i = 0; i <= n / 25; i++){
            int rest = n - i * 25;
            int a = rest / 10;
            int b = (rest % 10) / 5;
            result += (a + 1) * (a + b + 1);
        }
        return result % mod;
    }

    /**
     * 面试题51. 数组中的逆序对
     *
     * 在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组，求出这个数组中的逆序对的总数。
     *
     * @param nums
     * @return
     */
    public int reversePairs(int[] nums) {
        int length = nums.length;
        if(length < 2)
            return 0;
        int[] copy = new int[length];
        for(int i = 0; i < length; i++)
            copy[i] = nums[i];
        int[] temp = new int[length];
        return reversePairs(copy,0,length - 1,temp);
    }

    private int reversePairs(int[] nums,int left,int right,int[] temp){
        if(left == right)
            return 0;
        int mid = left + (right - left)/2;
        int leftPairs = reversePairs(nums,left,mid,temp);
        int rightPairs = reversePairs(nums,mid + 1, right,temp);
        if(nums[mid] <= nums[mid + 1]){
            return leftPairs + rightPairs;
        }
        int crossPairs = mergeAndCount(nums,left,mid,right,temp);
        return leftPairs + rightPairs + crossPairs;
    }

    private int mergeAndCount(int[] nums,int left ,int mid,int right ,int[] temp){
        for(int i = left; i <= right; i++){
            temp[i] = nums[i];
        }
        int i = left;
        int j = mid + 1;
        int count = 0;
        for(int k = left; k <= right;k++){
            if(i== mid + 1){
                nums[k] = temp[j];
                j++;
            }else if(j == right + 1){
                nums[k] = temp[i];
                i++;
            }else if(temp[i] <= temp[j]){
                nums[k] = temp[i];
                i++;
            }else {
                nums[k] = temp[j];
                j++;
                count += (mid - i + 1);
            }
        }
        return count;
    }

    public String randomStr(int n){
        Random r = new Random();
        StringBuilder strBuilder = new StringBuilder();
        for(int i = 0;i < n; i++){
            int number = r.nextInt(26);
            strBuilder.append((char)(number + (int)('A')));
        }
        return strBuilder.toString();
    }

    /**
     * 983. 最低票价
     * 在一个火车旅行很受欢迎的国度，你提前一年计划了一些火车旅行。在接下来的一年里，你要旅行的日子将以一个名为 days 的数组给出。每一项是一个从 1 到 365 的整数。
     *
     * 火车票有三种不同的销售方式：
     *
     * 一张为期一天的通行证售价为 costs[0] 美元；
     * 一张为期七天的通行证售价为 costs[1] 美元；
     * 一张为期三十天的通行证售价为 costs[2] 美元。
     * 通行证允许数天无限制的旅行。 例如，如果我们在第 2 天获得一张为期 7 天的通行证，那么我们可以连着旅行 7 天：第 2 天、第 3 天、第 4 天、第 5 天、第 6 天、第 7 天和第 8 天。
     *
     * 返回你想要完成在给定的列表 days 中列出的每一天的旅行所需要的最低消费。
     *
     * @param days
     * @param costs
     * @return
     */

    int[] costs;
    Integer[] memo;
    Set<Integer> dayset;
    public int mincostTickets(int[] days, int[] costs) {
        this.costs = costs;
        memo = new Integer[366];
        dayset = new HashSet<>();
        for(int d : days){
            dayset.add(d);
        }
        return mincostTicketsDP(1);
    }

    private int mincostTicketsDP(int i){
        if(i > 365)
            return 0;
        if(memo[i] != null)
            return memo[i];
        if(dayset.contains(i)){
            memo[i] = Math.min(Math.min(mincostTicketsDP(i + 1) + costs[0],mincostTicketsDP(i + 7) + costs[1]),mincostTicketsDP(i + 30) + costs[2]);
        }else{
            memo[i] = mincostTicketsDP(i + 1);
        }
        return memo[i];
    }

    /**
     * 实现 pow(x, n) ，即计算 x 的 n 次幂函数。
     * @param x
     * @param n
     * @return
     */
    public double myPow(double x, int n) {
        long N = n;
        return N >= 0 ? quickMulRecursive(x,N):1.0/quickMulRecursive(x,-N);
//        return N >= 0 ? quickMulIterate(x,N) : 1.0 / quickMulIterate(x, -N);
    }

    public double quickMulRecursive(double x,long N){
        if(N == 0)
            return 1.0;
        double y = quickMulRecursive(x,N / 2);
        return N % 2 == 0 ? y* y : y * y * x;
    }

    public double quickMulIterate(double x,long N){
        double result = 1.0;
        double x_contribute = x;
        while (N>0){
            if(N % 2 == 1){
                result *= x_contribute;
            }
            x_contribute *= x_contribute;
            N /= 2;
        }
        return result;
    }
}
