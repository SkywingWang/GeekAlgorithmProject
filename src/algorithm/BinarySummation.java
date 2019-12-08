package algorithm;
/**
 * created by Sven
 * on 2019-12-07
 *
 * 给定两个二进制字符串，返回他们的和（用二进制表示）。
 *
 * 输入为非空字符串且只包含数字 1 和 0。
 *
 */
public class BinarySummation {
    public String addBinary(String a, String b) {
        if(a == null || a.length() == 0)
            return b;
        if(b == null || b.length() == 0)
            return a;
        StringBuilder result = new StringBuilder();
        int i = 1;
        int isCarry = 0;
        while (a.length() - i >= 0 || b.length() - i >= 0){
            int sum = isCarry;
            if(a.length() - i >= 0)
                sum = sum + (a.charAt(a.length() -i) - '0');
            if(b.length() - i >= 0)
                sum = sum + (b.charAt(b.length() - i) - '0');
            isCarry = sum / 2;
            result.append(sum % 2);
            i++;
        }
        if(isCarry == 1)
            result.append(isCarry);
        return result.reverse().toString();
    }
}
