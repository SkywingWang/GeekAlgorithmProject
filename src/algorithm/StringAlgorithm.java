package algorithm;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/**
 * created by Sven
 * on 2019-12-14
 * <p>
 * 字符串String的相关算法
 */
public class StringAlgorithm {

    /**
     * created by Sven
     * on 2019-12-07
     * <p>
     * 给定两个二进制字符串，返回他们的和（用二进制表示）。
     * <p>
     * 输入为非空字符串且只包含数字 1 和 0。
     */
    public String addBinary(String a, String b) {
        if (a == null || a.length() == 0)
            return b;
        if (b == null || b.length() == 0)
            return a;
        StringBuilder result = new StringBuilder();
        int i = 1;
        int isCarry = 0;
        while (a.length() - i >= 0 || b.length() - i >= 0) {
            int sum = isCarry;
            if (a.length() - i >= 0)
                sum = sum + (a.charAt(a.length() - i) - '0');
            if (b.length() - i >= 0)
                sum = sum + (b.charAt(b.length() - i) - '0');
            isCarry = sum / 2;
            result.append(sum % 2);
            i++;
        }
        if (isCarry == 1)
            result.append(isCarry);
        return result.reverse().toString();
    }

    /**
     * created by Sven
     * on 2019-12-10
     * <p>
     * 给定字符串，返回字符串中最后一个单词的长度，空格隔开
     */
    public int lengthOfLastWord(String s) {
        if (s == null || s.length() == 0)
            return 0;
        int i = s.length() - 1;
        char space = new Character(' ');
        while (i >= 0) {
            if (s.charAt(i) == space) {
                i--;
            } else {
                int j = i;
                while (j >= 0) {
                    if (s.charAt(j) == space) {
                        return i - j;
                    } else {
                        j--;
                    }
                }
                return i - j;
            }
        }
        return 0;
    }


    /**
     * created by Sven
     * on 2019-12-10
     *
     * 给定字符串J 代表石头中宝石的类型，和字符串 S代表你拥有的石头。 S 中每个字符代表了一种你拥有的石头的类型，你想知道你拥有的石头中有多少是宝石。
     *
     * J 中的字母不重复，J 和 S中的所有字符都是字母。字母区分大小写，因此"a"和"A"是不同类型的石头。
     *
     */
    public int numJewelsInStones(String J, String S) {
        if(J == null || "".equals(J) || S == null || "".equals(S))
            return 0;
        char[] tmpJ = J.toCharArray();

        Set<Character> setJ = new HashSet<Character>();
        for(char c : tmpJ){
            setJ.add(c);
        }
        int count = 0;
        char[] tmpS = S.toCharArray();
        for(int i = 0; i < tmpS.length ;i++){
            if(setJ.contains(tmpS[i]))
                count ++;
        }
        return count;
    }

    /**
     *
     * 给你一个有效的 IPv4 地址 address，返回这个 IP 地址的无效化版本。
     *
     * 所谓无效化 IP 地址，其实就是用 "[.]" 代替了每个 "."。
     *
     * @param address
     * @return
     */
    public String defangIPaddr(String address) {
        if(address == null)
            return null;
        char[] addressChar = address.toCharArray();
        StringBuffer result = new StringBuffer();
        for(int i = 0; i < addressChar.length; i ++){
            if(addressChar[i] == '.'){
                result.append("[.]");
            }else{
                result.append(addressChar[i]);
            }
        }
        return result.toString();

    }

    /**
     * 验证回文串
     *
     * 给定一个字符串，验证它是否是回文串，只考虑字母和数字字符，可以忽略字母的大小写。
     *
     * 说明：本题中，我们将空字符串定义为有效的回文串。
     *
     * @param s
     * @return
     */
    public boolean isPalindrome(String s) {
        if(s == null || "".equals(s))
            return true;
        int l = 0, r = s.length() - 1;
        s = s.toUpperCase();
        while(l < r){
            if(!Character.isLetterOrDigit(s.charAt(l)))
                l ++;
            else if(!Character.isLetterOrDigit(s.charAt(r)))
                r --;
            else if(s.charAt(l) == s.charAt(r)){
                l++;
                r--;
            }else{
                return false;
            }

        }
        return true;
    }
}
