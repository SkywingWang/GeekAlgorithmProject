package algorithm;
/**
 * created by Sven
 * on 2019-12-07
 *
 * 加一
 *
 * 给定一个由整数组成的非空数组所表示的非负整数，在该数的基础上加一。
 *
 * 最高位数字存放在数组的首位， 数组中每个元素只存储单个数字。
 *
 * 你可以假设除了整数 0 之外，这个整数不会以零开头。
 *
 */
public class ArrayPlusOne {
    public int[] plusOne(int[] digits) {
        if(digits == null)
            return digits;
        int i = digits.length - 1;
        boolean isCarry = false;
        while (i > 0){
            if(isCarry){
                if(digits[i] + 1 == 10){
                    isCarry = true;
                    digits[i] = 0;
                }else{
                    digits[i] = digits[i] + 1;
                    isCarry = false;
                    break;
                }
            }else{
                if(digits[i] + 1 == 10){
                    isCarry = true;
                    digits[i] = 0;
                }else{
                    digits[i] = digits[i] + 1;
                    isCarry = false;
                    break;
                }
            }
            i--;
        }
        if(isCarry || digits.length == 1){
            if(digits[0] + 1 == 10){
                digits = new int[digits.length + 1];
                digits[0] = 1;
            }else{
                digits[0] = digits[0] + 1;
            }
        }
        return digits;
    }
}
