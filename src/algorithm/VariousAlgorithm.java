package algorithm;

import data.State;
import data.TreeNode;

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
        if ((y4 - y3) * (x2 - x1) == (y2 - y1) * (x4 - x3)) {
            // 判断两直线是否重叠
            if ((y2 - y1) * (x3 - x1) == (y3 - y1) * (x2 - x1)) {
                // 判断 (x3, y3) 是否在「线段」(x1, y1)~(x2, y2) 上
                if (isInside(x1, y1, x2, y2, x3, y3)) {
                    updateRes(ans, x3, y3);
                }
                // 判断 (x4, y4) 是否在「线段」(x1, y1)~(x2, y2) 上
                if (isInside(x1, y1, x2, y2, x4, y4)) {
                    updateRes(ans, (double) x4, (double) y4);
                }
                // 判断 (x1, y1) 是否在「线段」(x3, y3)~(x4, y4) 上
                if (isInside(x3, y3, x4, y4, x1, y1)) {
                    updateRes(ans, (double) x1, (double) y1);
                }
                // 判断 (x2, y2) 是否在「线段」(x3, y3)~(x4, y4) 上
                if (isInside(x3, y3, x4, y4, x2, y2)) {
                    updateRes(ans, (double) x2, (double) y2);
                }
            }
        } else {
            // 联立方程得到 t1 和 t2 的值
            double t1 = (double) (x3 * (y4 - y3) + y1 * (x4 - x3) - y3 * (x4 - x3) - x1 * (y4 - y3)) / ((x2 - x1) * (y4 - y3) - (x4 - x3) * (y2 - y1));
            double t2 = (double) (x1 * (y2 - y1) + y3 * (x2 - x1) - y1 * (x2 - x1) - x3 * (y2 - y1)) / ((x4 - x3) * (y2 - y1) - (x2 - x1) * (y4 - y3));
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
     *
     * @param n
     * @return
     */
    public int waysToChange(int n) {
        if (n <= 0)
            return 0;
        if (n < 5)
            return 1;
        int mod = 1000000007;
        int result = 0;
        for (int i = 0; i <= n / 25; i++) {
            int rest = n - i * 25;
            int a = rest / 10;
            int b = (rest % 10) / 5;
            result += (a + 1) * (a + b + 1);
        }
        return result % mod;
    }

    /**
     * 面试题51. 数组中的逆序对
     * <p>
     * 在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组，求出这个数组中的逆序对的总数。
     *
     * @param nums
     * @return
     */
    public int reversePairs(int[] nums) {
        int length = nums.length;
        if (length < 2)
            return 0;
        int[] copy = new int[length];
        for (int i = 0; i < length; i++)
            copy[i] = nums[i];
        int[] temp = new int[length];
        return reversePairs(copy, 0, length - 1, temp);
    }

    private int reversePairs(int[] nums, int left, int right, int[] temp) {
        if (left == right)
            return 0;
        int mid = left + (right - left) / 2;
        int leftPairs = reversePairs(nums, left, mid, temp);
        int rightPairs = reversePairs(nums, mid + 1, right, temp);
        if (nums[mid] <= nums[mid + 1]) {
            return leftPairs + rightPairs;
        }
        int crossPairs = mergeAndCount(nums, left, mid, right, temp);
        return leftPairs + rightPairs + crossPairs;
    }

    private int mergeAndCount(int[] nums, int left, int mid, int right, int[] temp) {
        for (int i = left; i <= right; i++) {
            temp[i] = nums[i];
        }
        int i = left;
        int j = mid + 1;
        int count = 0;
        for (int k = left; k <= right; k++) {
            if (i == mid + 1) {
                nums[k] = temp[j];
                j++;
            } else if (j == right + 1) {
                nums[k] = temp[i];
                i++;
            } else if (temp[i] <= temp[j]) {
                nums[k] = temp[i];
                i++;
            } else {
                nums[k] = temp[j];
                j++;
                count += (mid - i + 1);
            }
        }
        return count;
    }

    public String randomStr(int n) {
        Random r = new Random();
        StringBuilder strBuilder = new StringBuilder();
        for (int i = 0; i < n; i++) {
            int number = r.nextInt(26);
            strBuilder.append((char) (number + (int) ('A')));
        }
        return strBuilder.toString();
    }

    /**
     * 983. 最低票价
     * 在一个火车旅行很受欢迎的国度，你提前一年计划了一些火车旅行。在接下来的一年里，你要旅行的日子将以一个名为 days 的数组给出。每一项是一个从 1 到 365 的整数。
     * <p>
     * 火车票有三种不同的销售方式：
     * <p>
     * 一张为期一天的通行证售价为 costs[0] 美元；
     * 一张为期七天的通行证售价为 costs[1] 美元；
     * 一张为期三十天的通行证售价为 costs[2] 美元。
     * 通行证允许数天无限制的旅行。 例如，如果我们在第 2 天获得一张为期 7 天的通行证，那么我们可以连着旅行 7 天：第 2 天、第 3 天、第 4 天、第 5 天、第 6 天、第 7 天和第 8 天。
     * <p>
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
        for (int d : days) {
            dayset.add(d);
        }
        return mincostTicketsDP(1);
    }

    private int mincostTicketsDP(int i) {
        if (i > 365)
            return 0;
        if (memo[i] != null)
            return memo[i];
        if (dayset.contains(i)) {
            memo[i] = Math.min(Math.min(mincostTicketsDP(i + 1) + costs[0], mincostTicketsDP(i + 7) + costs[1]), mincostTicketsDP(i + 30) + costs[2]);
        } else {
            memo[i] = mincostTicketsDP(i + 1);
        }
        return memo[i];
    }

    /**
     * 实现 pow(x, n) ，即计算 x 的 n 次幂函数。
     *
     * @param x
     * @param n
     * @return
     */
    public double myPow(double x, int n) {
        long N = n;
        return N >= 0 ? quickMulRecursive(x, N) : 1.0 / quickMulRecursive(x, -N);
//        return N >= 0 ? quickMulIterate(x,N) : 1.0 / quickMulIterate(x, -N);
    }

    public double quickMulRecursive(double x, long N) {
        if (N == 0)
            return 1.0;
        double y = quickMulRecursive(x, N / 2);
        return N % 2 == 0 ? y * y : y * y * x;
    }

    public double quickMulIterate(double x, long N) {
        double result = 1.0;
        double x_contribute = x;
        while (N > 0) {
            if (N % 2 == 1) {
                result *= x_contribute;
            }
            x_contribute *= x_contribute;
            N /= 2;
        }
        return result;
    }

    /**
     * 面试题64. 求1+2+…+n
     *
     * 求 1+2+...+n ，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。
     *
     * @param n
     * @return
     */
    public int sumNums(int n) {
        if (n == 0)
            return 0;
        if (n == 1)
            return 1;
        return sumNums(n - 1) + n;

    }

    /**
     * 837. 新21点
     *
     * 爱丽丝参与一个大致基于纸牌游戏 “21点” 规则的游戏，描述如下：
     *
     * 爱丽丝以 0 分开始，并在她的得分少于 K 分时抽取数字。 抽取时，她从 [1, W] 的范围中随机获得一个整数作为分数进行累计，其中 W 是整数。 每次抽取都是独立的，其结果具有相同的概率。
     *
     * 当爱丽丝获得不少于 K 分时，她就停止抽取数字。 爱丽丝的分数不超过 N 的概率是多少？
     *
     * @param N
     * @param K
     * @param W
     * @return
     */
    public double new21Game(int N, int K, int W) {
        if(K == 0){
            return 1.0;
        }
        double[] dp = new double[K + W + 1];
        for(int i = K; i <= N && i < K + W; i++){
            dp[i] = 1.0;
        }
        dp[K - 1] = 1.0 * Math.min(N - K + 1, W) / W;
        for(int i = K - 2 ; i >= 0; i--){
            dp[i] = dp[i + 1] - (dp[i + W + 1] - dp[i + 1]) / W;
        }
        return dp[0];
    }

    /**
     * 9. 回文数
     *
     * 判断一个整数是否是回文数。回文数是指正序（从左向右）和倒序（从右向左）读都是一样的整数。
     *
     * @param x
     * @return
     */
    public boolean isPalindrome(int x) {
        if(x == 0)
            return true;
        if(x < 0 || (x % 10 == 0))
            return false;
        int revertedNumber = 0;
        while (x > revertedNumber){
            revertedNumber = revertedNumber * 10 + x % 10;
            x /= 10;
        }
        return x == revertedNumber || x == revertedNumber / 10;
    }

    /**
     *  面试题 16.11. 跳水板
     *
     *  你正在使用一堆木板建造跳水板。有两种类型的木板，其中长度较短的木板长度为shorter，长度较长的木板长度为longer。你必须正好使用k块木板。编写一个方法，生成跳水板所有可能的长度。
     *
     * 返回的长度需要从小到大排列。
     *
     * @param shorter
     * @param longer
     * @param k
     * @return
     */
    public int[] divingBoard(int shorter, int longer, int k) {
        if(k == 0)
            return new int[0];
        if(shorter == longer)
            return new int[]{shorter * k};
        int[] lengths = new int[k + 1];
        for(int i = 0;i<=k;i++){
            lengths[i] = shorter * (k - i) + longer * i;
        }
        return lengths;
    }

    /**
     * 1025. 除数博弈
     *
     * 爱丽丝和鲍勃一起玩游戏，他们轮流行动。爱丽丝先手开局。
     *
     * 最初，黑板上有一个数字 N 。在每个玩家的回合，玩家需要执行以下操作：
     *
     * 选出任一 x，满足 0 < x < N 且 N % x == 0 。
     * 用 N - x 替换黑板上的数字 N 。
     * 如果玩家无法执行这些操作，就会输掉游戏。
     *
     * 只有在爱丽丝在游戏中取得胜利时才返回 True，否则返回 false。假设两个玩家都以最佳状态参与游戏。
     *
     * @param N
     * @return
     */
    public boolean divisorGame(int N) {
        return N % 2 == 0;
    }

    /**
     * 343. 整数拆分
     * 给定一个正整数 n，将其拆分为至少两个正整数的和，并使这些整数的乘积最大化。 返回你可以获得的最大乘积。
     * @param n
     * @return
     */
    public int integerBreak(int n) {
        int[] dp = new int[n + 1];
        for(int i = 2; i <= n;i++){
            int curMax = 0;
            for(int j = 1; j < i; j++){
                curMax = Math.max(curMax,Math.max(j * (i - j),j * dp[i - j]));
            }
            dp[i] = curMax;
        }
        return dp[n];
    }

    /**
     * 207. 课程表
     *
     * 你这个学期必须选修 numCourse 门课程，记为 0 到 numCourse-1 。
     *
     * 在选修某些课程之前需要一些先修课程。 例如，想要学习课程 0 ，你需要先完成课程 1 ，我们用一个匹配来表示他们：[0,1]
     *
     * 给定课程总量以及它们的先决条件，请你判断是否可能完成所有课程的学习？
     *
     *  
     *
     * @param numCourses
     * @param prerequisites
     * @return
     */
    List<List<Integer>> edges;
    int[] visited;
    boolean valid = true;
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        edges = new ArrayList<List<Integer>>();
        for(int i = 0; i < numCourses; i++)
            edges.add(new ArrayList<Integer>());
        visited = new int[numCourses];
        for(int[] info : prerequisites)
            edges.get(info[1]).add(info[0]);
        for(int i = 0; i < numCourses && valid; i++){
            if(visited[i] == 0)
                dfsCanFinish(i);
        }
        return valid;
    }

    private void dfsCanFinish(int u){
        visited[u] = 1;
        for(int v:edges.get(u)){
            if(visited[v] == 0){
                dfsCanFinish(v);
                if(!valid)
                    return;
            }else if(visited[v] == 1){
                valid = false;
                return;
            }
        }
        visited[u] = 2;
    }

    /**
     * 剑指 Offer 10- II. 青蛙跳台阶问题
     *
     * 一只青蛙一次可以跳上1级台阶，也可以跳上2级台阶。求该青蛙跳上一个 n 级的台阶总共有多少种跳法。
     *
     * 答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。
     *
     *
     * @param n
     * @return
     */
    public int numWays(int n) {
        int a = 1, b = 1, sum;
        for(int i = 0; i < n;i++){
            sum = (a + b) % 1000000007;
            a = b;
            b = sum;
        }
        return a;
    }

    /**
     * 546. 移除盒子
     *
     * 给出一些不同颜色的盒子，盒子的颜色由数字表示，即不同的数字表示不同的颜色。
     * 你将经过若干轮操作去去掉盒子，直到所有的盒子都去掉为止。每一轮你可以移除具有相同颜色的连续 k 个盒子（k >= 1），这样一轮之后你将得到 k*k 个积分。
     * 当你将所有盒子都去掉之后，求你能获得的最大积分和。
     *
     * @param boxes
     * @return
     */
    public int removeBoxes(int[] boxes) {
        int[][][] dp = new int[100][100][100];
        return calculatePointsBoxs(boxes,dp,0,boxes.length - 1,0);
    }
    private int calculatePointsBoxs(int[] boxes,int [][][] dp,int l,int r,int k){
        if(l > r) return 0;
        if(dp[l][r][k] != 0) return dp[l][r][k];
        while (r > l && boxes[r] == boxes[r - 1]){
            r--;
            k++;
        }
        dp[l][r][k] = calculatePointsBoxs(boxes, dp, l, r - 1, 0) + (k + 1) * (k + 1);
        for (int i = l; i < r; i++) {
            if (boxes[i] == boxes[r]) {
                dp[l][r][k] = Math.max(dp[l][r][k], calculatePointsBoxs(boxes, dp, l, i, k + 1) + calculatePointsBoxs(boxes, dp, i + 1, r - 1, 0));
            }
        }
        return dp[l][r][k];
    }


    /**
     * 529. 扫雷游戏
     * 让我们一起来玩扫雷游戏！
     *
     * 给定一个代表游戏板的二维字符矩阵。 'M' 代表一个未挖出的地雷，'E' 代表一个未挖出的空方块，'B' 代表没有相邻（上，下，左，右，和所有4个对角线）地雷的已挖出的空白方块，数字（'1' 到 '8'）表示有多少地雷与这块已挖出的方块相邻，'X' 则表示一个已挖出的地雷。
     *
     * 现在给出在所有未挖出的方块中（'M'或者'E'）的下一个点击位置（行和列索引），根据以下规则，返回相应位置被点击后对应的面板：
     *
     * 如果一个地雷（'M'）被挖出，游戏就结束了- 把它改为 'X'。
     * 如果一个没有相邻地雷的空方块（'E'）被挖出，修改它为（'B'），并且所有和其相邻的未挖出方块都应该被递归地揭露。
     * 如果一个至少与一个地雷相邻的空方块（'E'）被挖出，修改它为数字（'1'到'8'），表示相邻地雷的数量。
     * 如果在此次点击中，若无更多方块可被揭露，则返回面板。
     *
     *
     * @param board
     * @param click
     * @return
     */
    int[] updateBoardDirX = {0,1,0,-1,1,1,-1,-1};
    int[] updateBoardDirY = {1,0,-1,0,1,-1,1,-1};
    public char[][] updateBoard(char[][] board,int[] click){
        int x = click[0], y = click[1];
        if(board[x][y] == 'M')
            board[x][y] = 'X';
        else
            boardDFS(board,x,y);
        return board;
    }

    public void boardDFS(char[][] board,int x,int y){
        int cnt = 0;
        for(int i = 0; i < 8; i++){
            int tx = x + updateBoardDirX[i];
            int ty = y + updateBoardDirY[i];
            if(tx < 0 || tx >= board.length || ty < 0 || ty >= board[0].length)
                continue;
            if(board[tx][ty] == 'M')
                cnt++;
        }
        if(cnt > 0)
            board[x][y] = (char)(cnt + '0');
        else{
            board[x][y] = 'B';
            for(int i = 0; i < 8; i++){
                int tx = x + updateBoardDirX[i];
                int ty = y + updateBoardDirY[i];
                if (tx < 0 || tx >= board.length || ty < 0 || ty >= board[0].length || board[tx][ty] != 'E') {
                    continue;
                }
                boardDFS(board,tx,ty);
            }
        }

    }

    /**
     * 679. 24 点游戏
     * 你有 4 张写有 1 到 9 数字的牌。你需要判断是否能通过 *，/，+，-，(，) 的运算得到 24。
     * @param nums
     * @return
     */
    static final int TARGET = 24;
    static final double EPSILON = 1e-6;
    static final int ADD = 0, MULTIPLY = 1, SUBTRACT = 2, DIVIDE = 3;
    public boolean judgePoint24(int[] nums) {
        List<Double> list = new ArrayList<>();
        for(int num : nums)
            list.add((double) num);
        return solveJudgePoint24(list);
    }
    public boolean solveJudgePoint24(List<Double> list){
        if(list.size() == 0)
            return false;
        if(list.size() == 1)
            return Math.abs(list.get(0) - TARGET ) < EPSILON;
        int size = list.size();
        for(int i = 0; i < size ; i++){
            for (int j = 0; j < size; j++) {
                if (i != j) {
                    List<Double> list2 = new ArrayList<Double>();
                    for (int k = 0; k < size; k++) {
                        if (k != i && k != j) {
                            list2.add(list.get(k));
                        }
                    }
                    for (int k = 0; k < 4; k++) {
                        if (k < 2 && i > j) {
                            continue;
                        }
                        if (k == ADD) {
                            list2.add(list.get(i) + list.get(j));
                        } else if (k == MULTIPLY) {
                            list2.add(list.get(i) * list.get(j));
                        } else if (k == SUBTRACT) {
                            list2.add(list.get(i) - list.get(j));
                        } else if (k == DIVIDE) {
                            if (Math.abs(list.get(j)) < EPSILON) {
                                continue;
                            } else {
                                list2.add(list.get(i) / list.get(j));
                            }
                        }
                        if (solveJudgePoint24(list2)) {
                            return true;
                        }
                        list2.remove(list2.size() - 1);
                    }
                }
            }

        }
        return false;
    }

    /**
     * 201. 数字范围按位与
     *
     * 给定范围 [m, n]，其中 0 <= m <= n <= 2147483647，返回此范围内所有数字的按位与（包含 m, n 两端点）。
     *
     * @param m
     * @param n
     * @return
     */
    public int rangeBitwiseAnd(int m, int n) {
        int shift = 0;
        while (m < n){
            m >>= 1;
            n >>= 1;
            ++shift;
        }
        return m << shift;
    }

    /**
     * 841. 钥匙和房间
     *
     * 有 N 个房间，开始时你位于 0 号房间。每个房间有不同的号码：0，1，2，...，N-1，并且房间里可能有一些钥匙能使你进入下一个房间。
     *
     * 在形式上，对于每个房间 i 都有一个钥匙列表 rooms[i]，每个钥匙 rooms[i][j] 由 [0,1，...，N-1] 中的一个整数表示，其中 N = rooms.length。 钥匙 rooms[i][j] = v 可以打开编号为 v 的房间。
     *
     * 最初，除 0 号房间外的其余所有房间都被锁住。
     *
     * 你可以自由地在房间之间来回走动。
     *
     * 如果能进入每个房间返回 true，否则返回 false。
     *
     * @param rooms
     * @return
     */
    public boolean canVisitAllRooms(List<List<Integer>> rooms) {
        int n = rooms.size(),num = 0;
        boolean[] vis = new boolean[n];
        Queue<Integer> que = new LinkedList<>();
        vis[0] = true;
        que.offer(0);
        while(!que.isEmpty()){
            int x = que.poll();
            num++;
            for(int it : rooms.get(x)){
                if(!vis[it]){
                    vis[it] = true;
                    que.offer(it);
                }
            }
        }
        return num == n;
    }

    /**
     * 486. 预测赢家
     *
     * 给定一个表示分数的非负整数数组。 玩家 1 从数组任意一端拿取一个分数，随后玩家 2 继续从剩余数组任意一端拿取分数，然后玩家 1 拿，…… 。每次一个玩家只能拿取一个分数，分数被拿取之后不再可取。直到没有剩余分数可取时游戏结束。最终获得分数总和最多的玩家获胜。
     * 给定一个表示分数的数组，预测玩家1是否会成为赢家。你可以假设每个玩家的玩法都会使他的分数最大化。
     * @param nums
     * @return
     */
    public boolean PredictTheWinner(int[] nums) {
        if(nums == null)
            return false;
        int length = nums.length;
        int[][] dp = new int[length][length];
        for(int i = 0; i < length; i ++){
            dp[i][i] = nums[i];
        }
        for(int i = length - 2; i >= 0; i--){
            for(int j = i + 1; j < length; j++){
                dp[i][j] = Math.max(nums[i] - dp[i + 1][j],nums[j] - dp[i][j-1]);
            }
        }
        return dp[0][length - 1] >= 0;
    }

    /**
     * 51. N 皇后
     * n 皇后问题研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。
     * 上图为 8 皇后问题的一种解法。
     * 给定一个整数 n，返回所有不同的 n 皇后问题的解决方案。
     * 每一种解法包含一个明确的 n 皇后问题的棋子放置方案，该方案中 'Q' 和 '.' 分别代表了皇后和空位。
     * @param n
     * @return
     */
    public List<List<String>> solveNQueens(int n) {
        List<List<String>> solutions = new ArrayList<>();
        int[] queens = new int[n];
        Arrays.fill(queens,-1);
        Set<Integer> columns = new HashSet<Integer>();
        Set<Integer> diagonals1 = new HashSet<Integer>();
        Set<Integer> diagonals2 = new HashSet<Integer>();
        backtrackNQueens(solutions, queens, n, 0, columns, diagonals1, diagonals2);
        return solutions;
    }

    private void backtrackNQueens(List<List<String>> solutions,int[] queens,int n, int row,Set<Integer> columns,Set<Integer> diagonals1, Set<Integer> diagonals2) {
        if(row == n){
            List<String> board = generateBoardNQueens(queens,n);
            solutions.add(board);
        }else{
            for(int i =0; i < n; i++){
                if(columns.contains(i)){
                    continue;
                }
                int diagonal1 = row - i;
                if(diagonals1.contains(diagonal1)){
                    continue;
                }
                int diagonal2 = row + i;
                if(diagonals2.contains(diagonal2)){
                    continue;
                }
                queens[row] = i;
                columns.add(i);
                diagonals1.add(diagonal1);
                diagonals2.add(diagonal2);
                backtrackNQueens(solutions,queens,n,row + 1,columns,diagonals1,diagonals2);
                queens[row] = -1;
                columns.remove(i);
                diagonals1.remove(diagonal1);
                diagonals2.remove(diagonal2);

            }
        }
    }

    private List<String> generateBoardNQueens(int[] queens,int n){
        List<String> board = new ArrayList<>();
        for(int i = 0; i < n; i++){
            char[] row = new char[n];
            Arrays.fill(row,'.');
            row[queens[i]] = 'Q';
            board.add(new String(row));

        }
        return board;
    }

    /**
     * 60. 第k个排列
     * 给出集合 [1,2,3,…,n]，其所有元素共有 n! 种排列。
     *
     * 按大小顺序列出所有排列情况，并一一标记，当 n = 3 时, 所有排列如下：
     *
     * "123"
     * "132"
     * "213"
     * "231"
     * "312"
     * "321"
     * 给定 n 和 k，返回第 k 个排列。
     *
     * 说明：
     *
     * 给定 n 的范围是 [1, 9]。
     * 给定 k 的范围是[1,  n!]。
     *
     * @param n
     * @param k
     * @return
     */
    public String getPermutation(int n, int k) {
        int[] factorial = new int[n];
        factorial[0] = 1;
        for(int i = 1; i < n; i++){
            factorial[i] = factorial[i - 1] * i;
        }
        k--;
        StringBuffer ans = new StringBuffer();
        int[] valid = new int[n + 1];
        Arrays.fill(valid,1);
        for(int i = 1; i <= n;i++) {
            int order = k / factorial[n - i] + 1;
            for(int j = 1; j <= n; j++){
                order -= valid[j];
                if(order == 0){
                    ans.append(j);
                    valid[j] = 0;
                    break;
                }
            }
            k %= factorial[n - i];
        }
        return ans.toString();
    }

    /**
     * 347. 前 K 个高频元素
     *
     * 给定一个非空的整数数组，返回其中出现频率前 k 高的元素。
     *
     * @param nums
     * @param k
     * @return
     */
    public int[] topKFrequent(int[] nums, int k) {
        Map<Integer,Integer> occurrences = new HashMap<>();
        for(int num : nums){
            occurrences.put(num,occurrences.getOrDefault(num,0) + 1);
        }
        PriorityQueue<int[]> queue = new PriorityQueue<>(new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[1] - o2[1];
            }
        });
        for(Map.Entry<Integer, Integer> entry : occurrences.entrySet()){
            int num = entry.getKey(), count = entry.getValue();
            if(queue.size() == k){
                if(queue.peek()[1] < count){
                    queue.poll();
                    queue.offer(new int[]{num,count});
                }
            }else {
                queue.offer(new int[]{num,count});
            }
        }
        int[] ret = new int[k];
        for(int i = 0; i < k; i++){
            ret[i] = queue.poll()[0];
        }
        return ret;
    }

    /**
     * 77. 组合
     *
     * 给定两个整数 n 和 k，返回 1 ... n 中所有可能的 k 个数的组合。
     *
     * @param n
     * @param k
     * @return
     */
    public List<List<Integer>> combine(int n, int k) {
        List<Integer> temp = new ArrayList<>();
        List<List<Integer>> ans = new ArrayList<>();
        for(int i = 1; i <= k; i++){
            temp.add(i);
        }
        temp.add(n + 1);
        int j = 0;
        while(j < k){
            ans.add(new ArrayList<>(temp.subList(0,k)));
            j = 0;
            while (j < k && temp.get(j) + 1 == temp.get(j + 1)){
                temp.set(j,j + 1);
                j++;
            }
            temp.set(j,temp.get(j) + 1);
        }
        return ans;
    }

    /**
     * 216. 组合总和 III
     *
     * 找出所有相加之和为 n 的 k 个数的组合。组合中只允许含有 1 - 9 的正整数，并且每种组合中不存在重复的数字。
     *
     * 说明：
     *
     * 所有数字都是正整数。
     * 解集不能包含重复的组合。 
     *
     * @param k
     * @param n
     * @return
     */
    List<Integer> tempCombinationSum3 = new ArrayList<>();
    List<List<Integer>> ansCombinationSum3 = new ArrayList<>();
    public List<List<Integer>> combinationSum3(int k, int n) {
        for (int mask = 0; mask < (1 << 9); ++mask) {
            if (check(mask, k, n)) {
                ansCombinationSum3.add(new ArrayList<Integer>(tempCombinationSum3));
            }
        }
        return ansCombinationSum3;
    }
    public boolean check(int mask,int k,int n){
        tempCombinationSum3.clear();
        for(int i = 0; i < 9; i ++){
            if (((1 << i) & mask) != 0) {
                tempCombinationSum3.add(i + 1);
            }
        }
        if(tempCombinationSum3.size() != k){
            return false;
        }
        int sum = 0;
        for(int num : tempCombinationSum3){
            sum += num;
        }
        return sum == n;
    }

    /**
     * 637. 二叉树的层平均值
     * 给定一个非空二叉树, 返回一个由每层节点平均值组成的数组。
     * @param root
     * @return
     */
    public List<Double> averageOfLevels(TreeNode root) {
        List<Double> averages = new ArrayList<>();
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()){
            double sum = 0;
            int size = queue.size();
            for(int i = 0; i < size; i++){
                TreeNode node = queue.poll();
                sum += node.val;
                TreeNode left = node.left, right = node.right;
                if(left != null){
                    queue.offer(left);
                }
                if(right != null){
                    queue.offer(right);
                }
            }
            averages.add(sum /size);
        }
        return averages;
    }

    /**
     * 37. 解数独
     *
     * 编写一个程序，通过已填充的空格来解决数独问题。
     *
     * 一个数独的解法需遵循如下规则：
     *
     * 数字 1-9 在每一行只能出现一次。
     * 数字 1-9 在每一列只能出现一次。
     * 数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。
     * 空白格用 '.' 表示。
     *
     * @param board
     */
    private boolean[][] lineSudoku = new boolean[9][9];
    private boolean[][] columnSudoku = new boolean[9][9];
    private boolean[][][] blockSudoku = new boolean[3][3][9];
    private boolean validSudoku = false;
    private List<int[]> spacesSudoku = new ArrayList<>();
    public void solveSudoku(char[][] board) {
        for(int i = 0; i < 9; i++){
            for(int j = 0; j < 9; j++){
                if(board[i][j] == '.'){
                    spacesSudoku.add(new int[]{i,j});
                }else{
                    int digit = board[i][j] - '0' - 1;
                    lineSudoku[i][digit] = columnSudoku[j][digit] = blockSudoku[i / 3][j / 3][digit] = true;
                }
            }
        }
        dfsSudoku(board,0);
    }

    private void dfsSudoku(char[][]board,int pos){
        if(pos == spacesSudoku.size()){
            validSudoku = true;
            return;
        }
        int[] space = spacesSudoku.get(pos);
        int i = space[0], j = space[1];
        for(int digit = 0; digit < 9 && !validSudoku;digit++){
            if(!lineSudoku[i][digit] && !columnSudoku[j][digit] && !blockSudoku[i / 3][j / 3][digit]){
                lineSudoku[i][digit] = columnSudoku[j][digit] = blockSudoku[i / 3][j / 3][digit] = true;
                board[i][j] = (char) ( digit + '0' + 1);
                dfsSudoku(board,pos + 1);
                lineSudoku[i][digit] = columnSudoku[j][digit] = blockSudoku[i / 3][j / 3][digit] = false;
            }
        }
    }

    /**
     * 52. N皇后 II
     * n 皇后问题研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。
     * @param n
     * @return
     */
    public int totalNQueens(int n) {
        return solveTotalNQueens(n,0,0,0,0);
    }
    private int solveTotalNQueens(int n, int row, int columns, int diagonals1, int diagonals2) {
        if (row == n) {
            return 1;
        } else {
            int count = 0;
            int availablePositions = ((1 << n) - 1) & (~(columns | diagonals1 | diagonals2));
            while (availablePositions != 0) {
                int position = availablePositions & (-availablePositions);
                availablePositions = availablePositions & (availablePositions - 1);
                count += solveTotalNQueens(n, row + 1, columns | position, (diagonals1 | position) << 1, (diagonals2 | position) >> 1);
            }
            return count;
        }
    }

    /**
     * 1024. 视频拼接
     *
     * 你将会获得一系列视频片段，这些片段来自于一项持续时长为 T 秒的体育赛事。这些片段可能有所重叠，也可能长度不一。
     *
     * 视频片段 clips[i] 都用区间进行表示：开始于 clips[i][0] 并于 clips[i][1] 结束。我们甚至可以对这些片段自由地再剪辑，例如片段 [0, 7] 可以剪切成 [0, 1] + [1, 3] + [3, 7] 三部分。
     *
     * 我们需要将这些片段进行再剪辑，并将剪辑后的内容拼接成覆盖整个运动过程的片段（[0, T]）。返回所需片段的最小数目，如果无法完成该任务，则返回 -1 。
     *
     * @param clips
     * @param T
     * @return
     */
    public int videoStitching(int[][] clips, int T) {
        int[] dp = new int[T + 1];
        Arrays.fill(dp,Integer.MAX_VALUE - 1);
        dp[0] = 0;
        for(int i = 1; i <= T; i++){
            for(int[] clip:clips){
                if(clip[0] < i && i <= clip[1]){
                    dp[i] = Math.min(dp[i],dp[clip[0]] + 1);
                }
            }
        }
        return dp[T] == Integer.MAX_VALUE - 1 ? -1 : dp[T];
    }

    /**
     *
     * 463. 岛屿的周长
     *
     * 给定一个包含 0 和 1 的二维网格地图，其中 1 表示陆地 0 表示水域。
     *
     * 网格中的格子水平和垂直方向相连（对角线方向不相连）。整个网格被水完全包围，但其中恰好有一个岛屿（或者说，一个或多个表示陆地的格子相连组成的岛屿）。
     *
     * 岛屿中没有“湖”（“湖” 指水域在岛屿内部且不和岛屿周围的水相连）。格子是边长为 1 的正方形。网格为长方形，且宽度和高度均不超过 100 。计算这个岛屿的周长。
     *
     * @param grid
     * @return
     */
    public int islandPerimeter(int[][] grid) {
        if(grid == null || grid.length == 0 || grid[0].length == 0){
            return 0;
        }
        int perimeter = 0;
        for(int i = 0; i < grid.length; i++){
            for(int j = 0; j < grid[0].length; j++){
                if(grid[i][j] == 1){
                    perimeter += 4;
                    if(i > 0 && grid[i-1][j] == 1){
                        perimeter-=2;
                    }
                    if(j > 0 && grid[i][j - 1] == 1){
                        perimeter-=2;
                    }
                }
            }
        }
        return perimeter;
    }

    /**
     * 140. 单词拆分 II
     *
     * 给定一个非空字符串 s 和一个包含非空单词列表的字典 wordDict，在字符串中增加空格来构建一个句子，使得句子中所有的单词都在词典中。返回所有这些可能的句子。
     *
     * 说明：
     *
     * 分隔时可以重复使用字典中的单词。
     * 你可以假设字典中没有重复的单词。
     *
     * @param s
     * @param wordDict
     * @return
     */
    public List<String> wordBreak(String s, List<String> wordDict) {
        Map<Integer,List<List<String>>> map = new HashMap<>();
        List<List<String>> wordBreaks = backtrack(s,s.length(),new HashSet<String>(wordDict),0,map);
        List<String> breakList = new LinkedList<>();
        for(List<String> wordBreak:wordBreaks){
            breakList.add(String.join(" ",wordBreak));
        }
        return breakList;
    }

    private List<List<String>> backtrack(String s, int length, Set<String> wordSet, int index, Map<Integer, List<List<String>>> map) {
        if(!map.containsKey(index)){
            List<List<String>> wordBreaks = new LinkedList<List<String>>();
            if(index == length){
                wordBreaks.add(new LinkedList<>());
            }
            for(int i = index + 1; i <= length;i++){
                String word = s.substring(index,i);
                if(wordSet.contains(word)){
                    List<List<String>> nextWordBreaks = backtrack(s,length,wordSet,i,map);
                    for(List<String> nextWordBreak : nextWordBreaks){
                        LinkedList<String> wordBreak = new LinkedList<>(nextWordBreak);
                        wordBreak.offerFirst(word);
                        wordBreaks.add(wordBreak);
                    }
                }
            }
            map.put(index,wordBreaks);
        }
        return map.get(index);
    }

    /**
     * 973. 最接近原点的 K 个点
     *
     * 我们有一个由平面上的点组成的列表 points。需要从中找出 K 个距离原点 (0, 0) 最近的点。
     *
     * （这里，平面上两点之间的距离是欧几里德距离。）
     *
     * 你可以按任何顺序返回答案。除了点坐标的顺序之外，答案确保是唯一的。
     *
     *
     * @param points
     * @param K
     * @return
     */
    public int[][] kClosest(int[][] points, int K) {
        Arrays.sort(points, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return (o1[0] * o1[0] + o1[1] * o1[1] - o2[0] * o2[0] - o2[1] * o2[1]);
            }
        });
        return Arrays.copyOfRange(points,0,K);
    }

    /**
     * 514. 自由之路
     *
     *
     * @param ring
     * @param key
     * @return
     */
    public int findRotateSteps(String ring, String key) {
        int n = ring.length(), m = key.length();
        List<Integer>[] pos = new List[26];
        for (int i = 0; i < 26; ++i) {
            pos[i] = new ArrayList<Integer>();
        }
        for (int i = 0; i < n; ++i) {
            pos[ring.charAt(i) - 'a'].add(i);
        }
        int[][] dp = new int[m][n];
        for (int i = 0; i < m; ++i) {
            Arrays.fill(dp[i], 0x3f3f3f);
        }
        for (int i : pos[key.charAt(0) - 'a']) {
            dp[0][i] = Math.min(i, n - i) + 1;
        }
        for (int i = 1; i < m; ++i) {
            for (int j : pos[key.charAt(i) - 'a']) {
                for (int k : pos[key.charAt(i - 1) - 'a']) {
                    dp[i][j] = Math.min(dp[i][j], dp[i - 1][k] + Math.min(Math.abs(j - k), n - Math.abs(j - k)) + 1);
                }
            }
        }
        return Arrays.stream(dp[m - 1]).min().getAsInt();
    }

    /**
     * 402. 移掉K位数字
     * 给定一个以字符串表示的非负整数 num，移除这个数中的 k 位数字，使得剩下的数字最小。
     *
     * 注意:
     *
     * num 的长度小于 10002 且 ≥ k。
     * num 不会包含任何前导零。
     *
     * @param num
     * @param k
     * @return
     */
    public String removeKdigits(String num, int k) {
        Deque<Character> deque = new LinkedList<>();
        int length = num.length();
        for(int i = 0; i < length;i++){
            char digit = num.charAt(i);
            while (!deque.isEmpty() && k > 0 && deque.peekLast() > digit){
                deque.pollLast();
                k--;
            }
            deque.offerLast(digit);
        }
        for(int i = 0; i < k; i++){
            deque.pollLast();
        }
        StringBuilder ret = new StringBuilder();
        boolean leadingZero = true;
        while (!deque.isEmpty()){
            char digit = deque.pollFirst();
            if(leadingZero && digit == '0'){
                continue;
            }
            leadingZero = false;
            ret.append(digit);
        }
        return ret.length() == 0 ? "0" : ret.toString();
    }

    /**
     * 1030. 距离顺序排列矩阵单元格
     *
     * 给出 R 行 C 列的矩阵，其中的单元格的整数坐标为 (r, c)，满足 0 <= r < R 且 0 <= c < C。
     *
     * 另外，我们在该矩阵中给出了一个坐标为 (r0, c0) 的单元格。
     *
     * 返回矩阵中的所有单元格的坐标，并按到 (r0, c0) 的距离从最小到最大的顺序排，其中，两单元格(r1, c1) 和 (r2, c2) 之间的距离是曼哈顿距离，|r1 - r2| + |c1 - c2|。（你可以按任何满足此条件的顺序返回答案。）
     *
     *
     * @param R
     * @param C
     * @param r0
     * @param c0
     * @return
     */
    public int[][] allCellsDistOrder(int R, int C, int r0, int c0) {
        int[][] ret = new int[R * C][];
        for(int i = 0; i < R; i++){
            for(int j = 0; j < C; j++){
                ret[i * C + j] = new int[]{i,j};
            }
        }
        Arrays.sort(ret, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return (Math.abs(o1[0] - r0) + Math.abs(o1[1] - c0)) - (Math.abs(o2[0] - r0) + Math.abs(o2[1] - c0));
            }
        });
        return ret;
    }

    /**
     * 134. 加油站
     * 在一条环路上有 N 个加油站，其中第 i 个加油站有汽油 gas[i] 升。
     *
     * 你有一辆油箱容量无限的的汽车，从第 i 个加油站开往第 i+1 个加油站需要消耗汽油 cost[i] 升。你从其中的一个加油站出发，开始时油箱为空。
     *
     * 如果你可以绕环路行驶一周，则返回出发时加油站的编号，否则返回 -1。
     *
     * 说明: 
     *
     * 如果题目有解，该答案即为唯一答案。
     * 输入数组均为非空数组，且长度相同。
     * 输入数组中的元素均为非负数。
     *
     * @param gas
     * @param cost
     * @return
     */
    public int canCompleteCircuit(int[] gas, int[] cost) {
        int n = gas.length;
        int i = 0;
        while(i < n){
            int sumOfGas = 0, sumOfCost = 0;
            int cnt = 0;
            while(cnt < n){
                int j = (i + cnt) % n;
                sumOfGas += gas[j];
                sumOfCost += cost[j];
                if(sumOfCost > sumOfGas){
                    break;
                }
                cnt++;
            }
            if(cnt == n){
                return i;
            }else{
                i = i + cnt + 1;
            }
        }
        return -1;
    }

    /**
     * 283. 移动零
     *
     * 给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。
     *
     * @param nums
     */
    public void moveZeroes(int[] nums) {
        if(nums == null){
            return;
        }
        int n = nums.length,left = 0,right = 0;
        while (right < n){
            if(nums[right] != 0){
                int temp = nums[left];
                nums[left] = nums[right];
                nums[right] = temp;
                left++;
            }
            right++;
        }
    }

    /**
     * 242. 有效的字母异位词
     *
     * 给定两个字符串 s 和 t ，编写一个函数来判断 t 是否是 s 的字母异位词。
     *
     * @param s
     * @param t
     * @return
     */
    public boolean isAnagram(String s, String t) {
        if(s.length() != t.length()){
            return false;
        }
        char[] str1 = s.toCharArray();
        char[] str2 = s.toCharArray();
        Arrays.sort(str1);
        Arrays.sort(str2);
        return Arrays.equals(str1,str2);
    }

    /**
     * 452. 用最少数量的箭引爆气球
     *
     * 在二维空间中有许多球形的气球。对于每个气球，提供的输入是水平方向上，气球直径的开始和结束坐标。由于它是水平的，所以纵坐标并不重要，因此只要知道开始和结束的横坐标就足够了。开始坐标总是小于结束坐标。
     *
     * 一支弓箭可以沿着 x 轴从不同点完全垂直地射出。在坐标 x 处射出一支箭，若有一个气球的直径的开始和结束坐标为 xstart，xend， 且满足  xstart ≤ x ≤ xend，则该气球会被引爆。可以射出的弓箭的数量没有限制。 弓箭一旦被射出之后，可以无限地前进。我们想找到使得所有气球全部被引爆，所需的弓箭的最小数量。
     *
     * 给你一个数组 points ，其中 points [i] = [xstart,xend] ，返回引爆所有气球所必须射出的最小弓箭数。
     *
     *
     * @param points
     * @return
     */
    public int findMinArrowShots(int[][] points) {
        if(points.length == 0){
            return 0;
        }
        Arrays.sort(points, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                if(o1[1] > o2[1]){
                    return 1;
                }else if(o1[1] < o2[1]){
                    return -1;
                }else{
                    return 0;
                }
            }
        });
        int pos = points[0][1];
        int ans = 1;
        for(int[] balloon : points){
            if(balloon[0] > pos){
                pos = balloon[1];
                ans++;
            }
        }
        return ans;
    }

    /**
     *
     * @param A
     * @param B
     * @param C
     * @param D
     * @return
     */
    public int fourSumCount(int[] A, int[] B, int[] C, int[] D) {
        Map<Integer,Integer> countAB = new HashMap<>();
        for(int u : A){
            for(int v : B){
                countAB.put(u + v,countAB.getOrDefault(u + v,0) + 1);
            }
        }
        int ans = 0;
        for(int u : C){
            for(int v : D){
                if(countAB.containsKey(-u-v)){
                    ans += countAB.get(-u-v);
                }
            }
        }
        return ans;
    }


    /**
     * 976. 三角形的最大周长
     * 给定由一些正数（代表长度）组成的数组 A，返回由其中三个长度组成的、面积不为零的三角形的最大周长。
     *
     * 如果不能形成任何面积不为零的三角形，返回 0。
     * @param A
     * @return
     */
    public int largestPerimeter(int[] A) {
        if(A==null || A.length < 3){
            return 0;
        }
        Arrays.sort(A);
        int indexR = A.length - 1;
        while (indexR > 1){
            if(A[indexR] < A[indexR-1] + A[indexR-2]){
                return A[indexR] + A[indexR-1] + A[indexR-2];
            }
            indexR --;
        }
        return 0;
    }

    /**
     * 321. 拼接最大数
     *
     * 给定长度分别为 m 和 n 的两个数组，其元素由 0-9 构成，表示两个自然数各位上的数字。现在从这两个数组中选出 k (k <= m + n) 个数字拼接成一个新的数，要求从同一个数组中取出的数字保持其在原数组中的相对顺序。
     *
     * 求满足该条件的最大数。结果返回一个表示该最大数的长度为 k 的数组。
     *
     * 说明: 请尽可能地优化你算法的时间和空间复杂度。
     *
     * @param nums1
     * @param nums2
     * @param k
     * @return
     */
    public int[] maxNumber(int[] nums1,int[] nums2,int k){
        int m = nums1.length, n = nums2.length;
        int[] maxSubsequence = new int[k];
        int start = Math.max(0,k - n),end = Math.min(k,m);
        for(int i = start; i <= end; i++){
            int[] subsequence1 = maxSubsequenceMaxNumber(nums1,i);
            int[] subsequence2 = maxSubsequenceMaxNumber(nums2,k-i);
            int[] curMaxSubsequence = mergeMaxNumber(subsequence1,subsequence2);
            if(compareMaxNumber(curMaxSubsequence,0,maxSubsequence,0) > 0){
                System.arraycopy(curMaxSubsequence,0,maxSubsequence,0,k);
            }
        }
        return maxSubsequence;
    }

    private int[] maxSubsequenceMaxNumber(int[] nums,int k){
        int length = nums.length;
        int[] stack = new int[k];
        int top = -1;
        int remain = length - k;
        for(int i = 0; i < length; i++){
            int num = nums[i];
            while (top >= 0 && stack[top] < num && remain > 0){
                top--;
                remain--;
            }
            if(top < k - 1){
                stack[++top] = num;
            }else {
                remain--;
            }
        }
        return stack;
    }
    private int[] mergeMaxNumber(int[] subsequence1,int[] subsequence2){
        int x = subsequence1.length, y = subsequence2.length;
        if(x == 0){
            return subsequence2;
        }
        if(y == 0){
            return subsequence1;
        }
        int mergeLength = x + y;
        int[] merged = new int[mergeLength];
        int index1 = 0, index2 = 0;
        for(int i = 0; i < mergeLength;i++){
            if(compareMaxNumber(subsequence1,index1,subsequence2,index2) > 0){
                merged[i] = subsequence1[index1++];
            }else{
                merged[i] = subsequence2[index2++];
            }
        }
        return merged;
    }

    private int compareMaxNumber(int[] subsequence1,int index1,int[] subsequence2,int index2){
        int x = subsequence1.length, y = subsequence2.length;
        while(index1 < x && index2 < y){
            int difference = subsequence1[index1] - subsequence2[index2];
            if(difference != 0){
                return difference;
            }
            index1++;
            index2++;
        }
        return (x - index1) - (y - index2);
    }

    /**
     * 621. 任务调度器
     *
     * 给你一个用字符数组 tasks 表示的 CPU 需要执行的任务列表。其中每个字母表示一种不同种类的任务。任务可以以任意顺序执行，并且每个任务都可以在 1 个单位时间内执行完。在任何一个单位时间，CPU 可以完成一个任务，或者处于待命状态。
     *
     * 然而，两个 相同种类 的任务之间必须有长度为整数 n 的冷却时间，因此至少有连续 n 个单位时间内 CPU 在执行不同的任务，或者在待命状态。
     *
     * 你需要计算完成所有任务所需要的 最短时间 。
     *
     *
     * @param tasks
     * @param n
     * @return
     */
    public int leastInterval(char[] tasks, int n) {
        Map<Character,Integer> freq = new HashMap<>();
        for(char ch:tasks){
            freq.put(ch,freq.getOrDefault(ch,0) + 1);
        }
        int m = freq.size();
        List<Integer> nextValid = new ArrayList<>();
        List<Integer> rest = new ArrayList<>();
        Set<Map.Entry<Character,Integer>> entrySet = freq.entrySet();
        for(Map.Entry<Character,Integer> entry : entrySet){
            int value = entry.getValue();
            nextValid.add(1);
            rest.add(value);
        }
        int time = 0;
        for(int i = 0; i < tasks.length; i++){
            time++;
            int minNextValid = Integer.MAX_VALUE;
            for(int j = 0; j < m; j++){
                if(rest.get(j) != 0){
                    minNextValid = Math.min(minNextValid,nextValid.get(j));
                }
            }
            time = Math.max(time,minNextValid);
            int best = -1;
            for(int j = 0; j < m; j++){
                if(rest.get(j)!=0 && nextValid.get(j) <= time){
                    if(best == -1 || rest.get(j) > rest.get(best)){
                        best = j;
                    }
                }
            }
            nextValid.set(best,time + n + 1);
            rest.set(best,rest.get(best) - 1);
        }
        return time;

    }

    /**
     * 62. 不同路径
     * 一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。
     *
     * 机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。
     *
     * 问总共有多少条不同的路径？
     *
     * @param m
     * @param n
     * @return
     */
    public int uniquePaths(int m, int n) {
        int[][] f = new int[m][n];
        for(int i = 0; i < m; i++){
            f[i][0] = 1;
        }
        for(int j = 0; j < n; j++){
            f[0][j] = 1;
        }
        for(int i = 1;i < m; i++){
            for(int j = 1;j < n;j++){
                f[i][j] = f[i - 1][j] + f[i][j-1];
            }
        }
        return f[m-1][n-1];
    }

    /**
     * 860. 柠檬水找零
     * 在柠檬水摊上，每一杯柠檬水的售价为 5 美元。
     *
     * 顾客排队购买你的产品，（按账单 bills 支付的顺序）一次购买一杯。
     *
     * 每位顾客只买一杯柠檬水，然后向你付 5 美元、10 美元或 20 美元。你必须给每个顾客正确找零，也就是说净交易是每位顾客向你支付 5 美元。
     *
     * 注意，一开始你手头没有任何零钱。
     *
     * 如果你能给每位顾客正确找零，返回 true ，否则返回 false 。
     *
     * @param bills
     * @return
     */
    public boolean lemonadeChange(int[] bills) {
        if(bills == null || bills.length == 0){
            return false;
        }
        int five = 0, ten = 0;
        for(int i = 0; i < bills.length; i++){
            if(bills[i] == 5){
                five++;
            }else if(bills[i] == 10){
                if(five > 0){
                    five--;
                    ten++;
                }else{
                    return false;
                }
            }else if(bills[i] == 20){
                if(ten > 0){
                    ten--;
                    if(five > 0){
                        five--;
                    }else{
                        return false;
                    }
                }else{
                    if(five > 2){
                        five -= 3;
                    }else {
                        return false;
                    }
                }
            }
        }
        return true;
    }
}
