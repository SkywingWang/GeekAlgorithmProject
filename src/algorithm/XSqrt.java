package algorithm;

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
public class XSqrt {
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
}
