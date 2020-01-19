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
}
