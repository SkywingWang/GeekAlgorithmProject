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
     *
     * 一个有名的按摩师会收到源源不断的预约请求，每个预约都可以选择接或不接。
     * 在每次预约服务之间要有休息时间，因此她不能接受相邻的预约。给定一个预约请求序列，
     * 替按摩师找到最优的预约集合（总预约时间最长），返回总的分钟数。
     *
     * 动态规划
     *
     * @param nums
     * @return
     */
    public int massage(int[] nums) {
        if(nums == null || nums.length == 0)
            return 0;
        if(nums.length == 1)
            return nums[0];

        if(nums.length == 2)
            return Math.max(nums[0],nums[1]);
        int[] result = new int[nums.length];
        result[0] = nums[0];
        result[1] = Math.max(nums[0],nums[1]);
        for(int i = 2; i < nums.length;i++){
            result[i] = Math.max(result[i - 1],result[i - 2] + nums[i]);
        }
        return result[nums.length - 1];
    }

    /**
     * 892. 三维形体的表面积
     *
     * 在 N * N 的网格上，我们放置一些 1 * 1 * 1  的立方体。
     *
     * 每个值 v = grid[i][j] 表示 v 个正方体叠放在对应单元格 (i, j) 上。
     *
     * 请你返回最终形体的表面积。
     *
     * @param grid
     * @return
     */
    public int surfaceArea(int[][] grid) {
        int[] dr = new int[]{0,1,0,-1};
        int[] dc = new int[]{1,0,-1,0};

        int length = grid.length;
        int result = 0;
        for(int r = 0;r<length;r++){
            for(int c=0;c < length;c++){
                if(grid[r][c] > 0){
                    result+=2;
                    for(int k = 0;k<4;k++){
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
        if(board == null || board.length == 0)
            return 0;
        // 先找到R
        int i = 0,j = 0,result = 0;
        boolean isFind = false;
        while (i < board.length){
            while (j < board[i].length){
                if(board[i][j] == 'R'){
                    isFind = true;
                    break;
                }
                j++;
            }
            if(isFind){
                break;
            }
            j = 0;
            i++;
        }
        int k = j + 1;
        while (k < board[i].length){

            if(board[i][k] == '.'){

            }else if(board[i][k] == 'B')
                break;
            else if(board[i][k] == 'p'){
                result++;
                break;
            }
            k++;
        }
        k = j - 1;
        while (k >= 0){
            if(board[i][k] == '.'){

            }else if(board[i][k] == 'B')
                break;
            if(board[i][k] == 'p'){
                result++;
                break;
            }
            k--;
        }
        k = i + 1;
        while (k < board.length){
            if(board[k][j] == '.'){

            }else if(board[k][j] == 'B')
                break;
            if(board[k][j] == 'p'){
                result++;
                break;
            }
            k ++;
        }
        k = i - 1;
        while (k >= 0){
            if(board[k][j] == '.'){

            }else if(board[k][j] == 'B')
                break;
            if(board[k][j] == 'p'){
                result++;
                break;
            }
            k --;
        }
        return result;
    }

}
