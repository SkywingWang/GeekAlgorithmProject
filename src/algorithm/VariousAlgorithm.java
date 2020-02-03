package algorithm;

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
     *
     * X的平方根
     *
     * 实现 int sqrt(int x) 函数。
     *
     * 计算并返回 x 的平方根，其中 x 是非负整数。
     *
     * 由于返回类型是整数，结果只保留整数的部分，小数部分将被舍去。
     *
     */

    public int mySqrt(int x) {
        if(x < 0)
            return 0;
        if(x < 2)
            return x;
        for(long i = 1; i <= x/2;i++){
            long l = (i + 1) * (i + 1);
            if(l > Integer.MAX_VALUE || l > x)
                return (int)i;
        }
        return 0;
    }

    /**
     * 牛顿法
     * @param x
     * @return
     */
    public int mySqrt2(int x){
        if(x < 0)
            return 0;
        long left = 0;
        long right = x /2 + 1;
        while(left < right){
            long mid = (left + right + 1) /2;
            long square = mid * mid;
            if(square > x){
                right = mid - 1;
            }else if(square < x){
                left = mid;
            }else {
                return (int)mid;
            }
        }
        return (int) left;
    }

    /**
     * 给你一个整数 n，请你帮忙计算并返回该整数「各位数字之积」与「各位数字之和」的差。
     *
     */
    public int subtractProductAndSum(int n) {
        if(n == 0)
            return 0;
        int count = 0;
        int product = 1;
        while(n/10 > 0){
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
     *
     * 给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
     *
     * 如果你最多只允许完成一笔交易（即买入和卖出一支股票），设计一个算法来计算你所能获取的最大利润。
     *
     * 注意你不能在买入股票前卖出股票。
     *
     * @param prices
     * @return
     */
    public int maxProfit(int[] prices) {
        int minPrice = Integer.MAX_VALUE;
        int spreadPrice = 0;
        if(prices == null || prices.length == 0)
            return 0;
        for(int i = 0; i < prices.length;i++){
            if(prices[i] < minPrice)
                minPrice = prices[i];
            else if(prices[i] - minPrice > spreadPrice){
                spreadPrice = prices[i] - minPrice;
            }
        }
        return spreadPrice;
    }

    /**
     * 买卖股票的最佳时机 II
     *
     * 给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
     *
     * 设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。
     *
     * 注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
     *
     *
     * @param prices
     * @return
     */
    public int maxProfit2(int[] prices) {
        boolean isBuy = false;
        int buyPrice = 0;
        int spreadPriceCount = 0;
        if(prices == null || prices.length == 0)
            return 0;
        for(int i = 0; i < prices.length - 1; i++){
            if(isBuy){
                if(prices[i] > prices[i + 1]){
                    spreadPriceCount = spreadPriceCount + prices[i] - buyPrice;
                    isBuy = false;
                }
            }else{
                if(prices[i] < prices[i+1]){
                    buyPrice = prices[i];
                    isBuy = true;
                }
            }
        }
        if(isBuy){
            spreadPriceCount = spreadPriceCount + prices[prices.length - 1] - buyPrice;
        }
        return spreadPriceCount;
    }

    /**
     * Excel表列名称
     *
     * 给定一个正整数，返回它在 Excel 表中相对应的列名称。
     *
     * @param n
     * @return
     */
    public String convertToTitle(int n) {
        if(n <=0)
            return null;
        StringBuffer resultStringBuffer = new StringBuffer();
        while (n > 0){
            int remainder = n % 26;
            if(remainder == 0){
                remainder = 26;
                n--;
            }
            resultStringBuffer.insert(0,(char)('A' + remainder - 1));
            n = n / 26;
        }
        return resultStringBuffer.toString();
    }

    /**
     * Excel表列序号
     *
     * 给定一个Excel表格中的列名称，返回其相应的列序号。
     *
     * @param s
     * @return
     */
    public int titleToNumber(String s) {
        if(s == null || s.length() == 0)
            return 0;
        int result = 0;
        for(int i = 0; i < s.length(); i++){
            int number = (int)(s.charAt(i) - 'A' + 1);
            result = result * 26 + number;
        }
        return result;
    }

    /**
     * 阶乘后的零
     *
     * 给定一个整数 n，返回 n! 结果尾数中零的数量。
     * @param n
     * @return
     */
    public int trailingZeroes(int n) {
        if(n < 5)
            return 0;
        long i = 5;
        int count = 0;
        while(i <= n){
            long N = i;
            while (N > 0){
                if(N % 5 == 0){
                    count ++;
                    N /= 5;
                }else{
                    break;
                }
            }
            i += 5;
        }

        return count;
    }

    /**
     * 颠倒二进制位
     *
     * 颠倒给定的 32 位无符号整数的二进制位。
     *
     * @param n
     * @return
     */
    public int reverseBits(int n) {
        int res = 0;
        int count = 0;
        while (count < 32){
            res <<= 1;
            res |= (n & 1);
            n >>= 1;
            count ++;
        }
        return res;
    }

    /**
     * 位1的个数
     *
     * 编写一个函数，输入是一个无符号整数，返回其二进制表达式中数字位数为 ‘1’ 的个数（也被称为汉明重量）。
     *
     * @param n
     * @return
     */
    public int hammingWeight(int n) {
        int count = 0;
        int mask = 1;
        for(int i = 0; i < 32; i ++){
            if((n & mask) != 0){
                count ++;
            }
            mask <<= 1;
        }
        return count;
    }

    /**
     * 快乐数
     *
     * 编写一个算法来判断一个数是不是“快乐数”。
     *
     * 一个“快乐数”定义为：对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和，然后重复这个过程直到这个数变为 1，也可能是无限循环但始终变不到 1。如果可以变为 1，那么这个数就是快乐数。
     *
     *
     * @param n
     * @return
     */
    public boolean isHappy(int n) {
        int slow = n,fast = n;
        do{
            slow = bitSquareSum(slow);
            fast = bitSquareSum(fast);
            fast = bitSquareSum(fast);
        }while (slow != fast);
        return slow == 1;
    }

    private int bitSquareSum(int n){
        int sum = 0;
        while (n > 0){
            int square = n % 10;
            sum += (square * square);
            n /= 10;
        }
        return sum;
    }
}
