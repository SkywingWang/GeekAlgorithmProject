package algorithm;

import java.util.*;

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

    /**
     * 205. 同构字符串
     *
     *
     * 给定两个字符串 s 和 t，判断它们是否是同构的。
     * 如果 s 中的字符可以被替换得到 t ，那么这两个字符串是同构的。
     * 所有出现的字符都必须用另一个字符替换，同时保留字符的顺序。两个字符不能映射到同一个字符上，但字符可以映射自己本身。
     * @param s
     * @param t
     * @return
     */
    public boolean isIsomorphic(String s, String t) {
        if(s == null && t == null)
            return true;
        if(s == null || t == null)
            return false;
        if(s.length() != t.length())
            return false;
        Map<Character,Character> sToTMap = new HashMap<>();
        Map<Character,Character> tToSMap = new HashMap<>();
        int length = s.length();
        for(int i = 0; i < length; i++){
            if(!sToTMap.containsKey(s.charAt(i)) && !tToSMap.containsKey(t.charAt(i))){
                sToTMap.put(s.charAt(i),t.charAt(i));
                tToSMap.put(t.charAt(i),s.charAt(i));
            }
            else if(!sToTMap.containsKey(s.charAt(i)) || !tToSMap.containsKey(t.charAt(i)))
                return false;
            else if(!sToTMap.get(s.charAt(i)).equals(t.charAt(i))){
                return false;
            }
        }
        return true;
    }

    /**
     * 1111. 有效括号的嵌套深度
     * 有效括号字符串 仅由 "(" 和 ")" 构成，并符合下述几个条件之一：
     *
     * 空字符串
     * 连接，可以记作 AB（A 与 B 连接），其中 A 和 B 都是有效括号字符串
     * 嵌套，可以记作 (A)，其中 A 是有效括号字符串
     * 类似地，我们可以定义任意有效括号字符串 s 的 嵌套深度 depth(S)：
     *
     * s 为空时，depth("") = 0
     * s 为 A 与 B 连接时，depth(A + B) = max(depth(A), depth(B))，其中 A 和 B 都是有效括号字符串
     * s 为嵌套情况，depth("(" + A + ")") = 1 + depth(A)，其中 A 是有效括号字符串
     * 例如：""，"()()"，和 "()(()())" 都是有效括号字符串，嵌套深度分别为 0，1，2，而 ")(" 和 "(()" 都不是有效括号字符串。
     *
     *  
     *
     * 给你一个有效括号字符串 seq，将其分成两个不相交的子序列 A 和 B，且 A 和 B 满足有效括号字符串的定义（注意：A.length + B.length = seq.length）。
     *
     * 现在，你需要从中选出 任意 一组有效括号字符串 A 和 B，使 max(depth(A), depth(B)) 的可能取值最小。
     *
     * 返回长度为 seq.length 答案数组 answer ，选择 A 还是 B 的编码规则是：如果 seq[i] 是 A 的一部分，那么 answer[i] = 0。否则，answer[i] = 1。即便有多个满足要求的答案存在，你也只需返回 一个。
     *
     *
     * @param seq
     * @return
     */
    public int[] maxDepthAfterSplit(String seq) {
        if(seq == null || seq.length() == 0)
            return null;
        int [] ans = new int[seq.length()];
        int index = 0;
        int s = 0;
        for(char c:seq.toCharArray()){
            ans[index++] = c == '(' ? ((s += 1) % 2): ((s -= 1) % 2);
        }
        return ans;
    }

}
