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
     * <p>
     * 给定字符串J 代表石头中宝石的类型，和字符串 S代表你拥有的石头。 S 中每个字符代表了一种你拥有的石头的类型，你想知道你拥有的石头中有多少是宝石。
     * <p>
     * J 中的字母不重复，J 和 S中的所有字符都是字母。字母区分大小写，因此"a"和"A"是不同类型的石头。
     */
    public int numJewelsInStones(String J, String S) {
        if (J == null || "".equals(J) || S == null || "".equals(S))
            return 0;
        char[] tmpJ = J.toCharArray();

        Set<Character> setJ = new HashSet<Character>();
        for (char c : tmpJ) {
            setJ.add(c);
        }
        int count = 0;
        char[] tmpS = S.toCharArray();
        for (int i = 0; i < tmpS.length; i++) {
            if (setJ.contains(tmpS[i]))
                count++;
        }
        return count;
    }

    /**
     * 给你一个有效的 IPv4 地址 address，返回这个 IP 地址的无效化版本。
     * <p>
     * 所谓无效化 IP 地址，其实就是用 "[.]" 代替了每个 "."。
     *
     * @param address
     * @return
     */
    public String defangIPaddr(String address) {
        if (address == null)
            return null;
        char[] addressChar = address.toCharArray();
        StringBuffer result = new StringBuffer();
        for (int i = 0; i < addressChar.length; i++) {
            if (addressChar[i] == '.') {
                result.append("[.]");
            } else {
                result.append(addressChar[i]);
            }
        }
        return result.toString();

    }

    /**
     * 验证回文串
     * <p>
     * 给定一个字符串，验证它是否是回文串，只考虑字母和数字字符，可以忽略字母的大小写。
     * <p>
     * 说明：本题中，我们将空字符串定义为有效的回文串。
     *
     * @param s
     * @return
     */
    public boolean isPalindrome(String s) {
        if (s == null || "".equals(s))
            return true;
        int l = 0, r = s.length() - 1;
        s = s.toUpperCase();
        while (l < r) {
            if (!Character.isLetterOrDigit(s.charAt(l)))
                l++;
            else if (!Character.isLetterOrDigit(s.charAt(r)))
                r--;
            else if (s.charAt(l) == s.charAt(r)) {
                l++;
                r--;
            } else {
                return false;
            }

        }
        return true;
    }

    /**
     * 205. 同构字符串
     * <p>
     * <p>
     * 给定两个字符串 s 和 t，判断它们是否是同构的。
     * 如果 s 中的字符可以被替换得到 t ，那么这两个字符串是同构的。
     * 所有出现的字符都必须用另一个字符替换，同时保留字符的顺序。两个字符不能映射到同一个字符上，但字符可以映射自己本身。
     *
     * @param s
     * @param t
     * @return
     */
    public boolean isIsomorphic(String s, String t) {
        if (s == null && t == null)
            return true;
        if (s == null || t == null)
            return false;
        if (s.length() != t.length())
            return false;
        Map<Character, Character> sToTMap = new HashMap<>();
        Map<Character, Character> tToSMap = new HashMap<>();
        int length = s.length();
        for (int i = 0; i < length; i++) {
            if (!sToTMap.containsKey(s.charAt(i)) && !tToSMap.containsKey(t.charAt(i))) {
                sToTMap.put(s.charAt(i), t.charAt(i));
                tToSMap.put(t.charAt(i), s.charAt(i));
            } else if (!sToTMap.containsKey(s.charAt(i)) || !tToSMap.containsKey(t.charAt(i)))
                return false;
            else if (!sToTMap.get(s.charAt(i)).equals(t.charAt(i))) {
                return false;
            }
        }
        return true;
    }

    /**
     * 1111. 有效括号的嵌套深度
     * 有效括号字符串 仅由 "(" 和 ")" 构成，并符合下述几个条件之一：
     * <p>
     * 空字符串
     * 连接，可以记作 AB（A 与 B 连接），其中 A 和 B 都是有效括号字符串
     * 嵌套，可以记作 (A)，其中 A 是有效括号字符串
     * 类似地，我们可以定义任意有效括号字符串 s 的 嵌套深度 depth(S)：
     * <p>
     * s 为空时，depth("") = 0
     * s 为 A 与 B 连接时，depth(A + B) = max(depth(A), depth(B))，其中 A 和 B 都是有效括号字符串
     * s 为嵌套情况，depth("(" + A + ")") = 1 + depth(A)，其中 A 是有效括号字符串
     * 例如：""，"()()"，和 "()(()())" 都是有效括号字符串，嵌套深度分别为 0，1，2，而 ")(" 和 "(()" 都不是有效括号字符串。
     * <p>
     *  
     * <p>
     * 给你一个有效括号字符串 seq，将其分成两个不相交的子序列 A 和 B，且 A 和 B 满足有效括号字符串的定义（注意：A.length + B.length = seq.length）。
     * <p>
     * 现在，你需要从中选出 任意 一组有效括号字符串 A 和 B，使 max(depth(A), depth(B)) 的可能取值最小。
     * <p>
     * 返回长度为 seq.length 答案数组 answer ，选择 A 还是 B 的编码规则是：如果 seq[i] 是 A 的一部分，那么 answer[i] = 0。否则，answer[i] = 1。即便有多个满足要求的答案存在，你也只需返回 一个。
     *
     * @param seq
     * @return
     */
    public int[] maxDepthAfterSplit(String seq) {
        if (seq == null || seq.length() == 0)
            return null;
        int[] ans = new int[seq.length()];
        int index = 0;
        int s = 0;
        for (char c : seq.toCharArray()) {
            ans[index++] = c == '(' ? ((s += 1) % 2) : ((s -= 1) % 2);
        }
        return ans;
    }

    /**
     * 8. 字符串转换整数 (atoi)
     * <p>
     * <p>
     * 请你来实现一个 atoi 函数，使其能将字符串转换成整数。
     * 首先，该函数会根据需要丢弃无用的开头空格字符，直到寻找到第一个非空格的字符为止。接下来的转化规则如下：
     * 如果第一个非空字符为正或者负号时，则将该符号与之后面尽可能多的连续数字字符组合起来，形成一个有符号整数。
     * 假如第一个非空字符是数字，则直接将其与之后连续的数字字符组合起来，形成一个整数。
     * 该字符串在有效的整数部分之后也可能会存在多余的字符，那么这些字符可以被忽略，它们对函数不应该造成影响。
     * 注意：假如该字符串中的第一个非空格字符不是一个有效整数字符、字符串为空或字符串仅包含空白字符时，则你的函数不需要进行转换，即无法进行有效转换。
     * 在任何情况下，若函数不能进行有效的转换时，请返回 0 。
     * 提示：
     * 本题中的空白字符只包括空格字符 ' ' 。
     * 假设我们的环境只能存储 32 位大小的有符号整数，那么其数值范围为 [−231,  231 − 1]。如果数值超过这个范围，请返回  INT_MAX (231 − 1) 或 INT_MIN (−231) 。
     *
     * @param str
     * @return
     */
    public int myAtoi(String str) {
        if (str == null || "".equals(str.trim()))
            return 0;
        str = str.trim();
        char[] chars = str.toCharArray();
        int index = 0;
        boolean negative = false;
        if (chars[index] == '-') {
            negative = true;
            index++;
        } else if (chars[index] == '+') {
            index++;
        } else if (!Character.isDigit(chars[index])) {
            return 0;
        }
        long result = 0;
        while (index < chars.length && Character.isDigit(chars[index])) {
            int digit = chars[index] - '0';
            result = result * 10 + digit;
            if (result > Integer.MAX_VALUE)
                return negative ? Integer.MIN_VALUE : Integer.MAX_VALUE;
            index++;
        }
        return negative ? -(int) result : (int) result;
    }

    /**
     * 151. 翻转字符串里的单词
     * 给定一个字符串，逐个翻转字符串中的每个单词。
     *
     * @param s
     * @return
     */
    public String reverseWords(String s) {
        if (s == null || s.trim().length() == 0)
            return "";
        s = s.trim();
        List<String> wordList = Arrays.asList(s.split("\\s+"));
        Collections.reverse(wordList);
        return String.join(" ", wordList);
    }

    public String reverseWords_2(String s) {
        if (s == null || s.trim().length() == 0)
            return "";
        s = s.trim();
        StringBuilder stringBuilder = new StringBuilder();
        Deque<String> d = new ArrayDeque<>();
        int left = 0, right = s.length() - 1;
        while (left <= right) {
            char c = s.charAt(left);
            if ((stringBuilder.length() != 0) && (c == ' ')) {
                d.offerFirst(stringBuilder.toString());
                stringBuilder.setLength(0);
            } else if (c != ' ') {
                stringBuilder.append(c);
            }
            ++left;
        }
        d.offerFirst(stringBuilder.toString());
        return String.join(" ", d);
    }

    /**
     * 466. 统计重复个数
     * 由 n 个连接的字符串 s 组成字符串 S，记作 S = [s,n]。例如，["abc",3]=“abcabcabc”。
     * <p>
     * 如果我们可以从 s2 中删除某些字符使其变为 s1，则称字符串 s1 可以从字符串 s2 获得。例如，根据定义，"abc" 可以从 “abdbec” 获得，但不能从 “acbbe” 获得。
     * <p>
     * 现在给你两个非空字符串 s1 和 s2（每个最多 100 个字符长）和两个整数 0 ≤ n1 ≤ 106 和 1 ≤ n2 ≤ 106。现在考虑字符串 S1 和 S2，其中 S1=[s1,n1] 、S2=[s2,n2] 。
     * <p>
     * 请你找出一个可以满足使[S2,M] 从 S1 获得的最大整数 M 。
     *
     * @param s1
     * @param n1
     * @param s2
     * @param n2
     * @return
     */
    public int getMaxRepetitions(String s1, int n1, String s2, int n2) {
        if (n1 == 0)
            return 0;
        char[] c1 = s1.toCharArray();
        char[] c2 = s2.toCharArray();
        int l1 = s1.length();
        int l2 = s2.length();
        int countS1 = 0; // 经历了多少个s1
        int countS2 = 0; // 经历了多少个s2
        int p = 0;         // 当前在s2的位置
        Map<Integer, int[]> mp = new HashMap<>(); //记录每一次s1 扫描结束后当前的状态，寻找循环
        while (countS1 < n1) {
            for (int i = 0; i < l1; i++) {
                if (c1[i] == c2[p]) {
                    p++;
                    if (p == l2) {
                        p = 0;
                        countS2++;
                    }
                }
            }
            countS1++;
            if (!mp.containsKey(p)) {
                mp.put(p, new int[]{countS1, countS2}); //记录当前状态
            } else {
                int[] last = mp.get(p);
                int circle1 = countS1 - last[0];
                int circle2 = countS2 - last[1];
                countS2 += circle2 * ((n1 - countS1) / circle1);
                countS1 = countS1 + ((n1 - countS1) / circle1) * circle1;
            }
        }
        return countS2 / n2;
    }

    /**
     * 面试题58 - II. 左旋转字符串
     * 字符串的左旋转操作是把字符串前面的若干个字符转移到字符串的尾部。请定义一个函数实现字符串左旋转操作的功能。比如，输入字符串"abcdefg"和数字2，该函数将返回左旋转两位得到的结果"cdefgab"。
     *
     * @param s
     * @param n
     * @return
     */
    public String reverseLeftWords(String s, int n) {
        if (s == null)
            return null;
        if (n > s.length())
            return s;
        return s.substring(n) + s.substring(0, n);
    }

    /**
     * 680. 验证回文字符串 Ⅱ
     * <p>
     * 给定一个非空字符串 s，最多删除一个字符。判断是否能成为回文字符串。
     *
     * @param s
     * @return
     */
    public boolean validPalindrome(String s) {
        if (s == null || s.length() == 0)
            return false;
        if (s.length() <= 2)
            return true;
        int left = 0, right = s.length() - 1;
        while (left < right) {
            char c1 = s.charAt(left), c2 = s.charAt(right);
            if (c1 == c2) {
                left++;
                right--;
            } else {
                boolean flag1 = true, flag2 = true;
                for (int i = left, j = right - 1; i < j; i++, j--) {
                    char c3 = s.charAt(i), c4 = s.charAt(j);
                    if (c3 != c4) {
                        flag1 = false;
                        break;
                    }
                }
                for (int i = left + 1, j = right; i < j; i++, j--) {
                    char c3 = s.charAt(i), c4 = s.charAt(j);
                    if (c3 != c4) {
                        flag2 = false;
                        break;
                    }
                }
                return flag1 || flag2;
            }
        }
        return true;
    }

    /**
     * 1371. 每个元音包含偶数次的最长子字符串
     * <p>
     * 给你一个字符串 s ，请你返回满足以下条件的最长子字符串的长度：每个元音字母，即 'a'，'e'，'i'，'o'，'u' ，在子字符串中都恰好出现了偶数次。
     *
     * @param s
     * @return
     */
    public int findTheLongestSubstring(String s) {
        if (s == null || s.length() == 0)
            return 0;
        int n = s.length();
        int[] pos = new int[1 << 5];
        Arrays.fill(pos, -1);
        int ans = 0, status = 0;
        pos[0] = 0;
        for (int i = 0; i < n; i++) {
            char ch = s.charAt(i);
            if (ch == 'a') {
                status ^= (1 << 0);
            } else if (ch == 'e') {
                status ^= (1 << 1);
            } else if (ch == 'i') {
                status ^= (1 << 2);
            } else if (ch == 'o') {
                status ^= (1 << 3);
            } else if (ch == 'u') {
                status ^= (1 << 4);
            }
            if (pos[status] >= 0) {
                ans = Math.max(ans, i + 1 - pos[status]);
            } else {
                pos[status] = i + 1;
            }
        }
        return ans;
    }

    /**
     * 5. 最长回文子串
     * 给定一个字符串 s，找到 s 中最长的回文子串。你可以假设 s 的最大长度为 1000。
     *
     * @param s
     * @return
     */
    public String longestPalindrome(String s) {
        if(s == null || s.length() < 1) return "";
        int start = 0, end = 0;
        for(int i = 0; i < s.length(); i++){
            int len1 = expandAroundCenter(s,i,i);
            int len2 = expandAroundCenter(s,i,i+1);
            int len = Math.max(len1,len2);
            if(len > end - start){
                start = i - (len - 1) / 2;
                end = i + len / 2;
            }
        }
        return s.substring(start, end + 1);
    }

    private int expandAroundCenter(String s, int left, int right) {
        int L = left, R = right;
        while (L >= 0 && R < s.length() && s.charAt(L) == s.charAt(R)) {
            L--;
            R++;
        }
        return R - L - 1;
    }

    /**
     * 76. 最小覆盖子串
     *
     * 给你一个字符串 S、一个字符串 T，请在字符串 S 里面找出：包含 T 所有字符的最小子串。
     *
     * @param s
     * @param t
     * @return
     */
    Map<Character, Integer> ori = new HashMap<Character, Integer>();
    Map<Character, Integer> cnt = new HashMap<Character, Integer>();
    public String minWindow(String s, String t) {
        int tLen = t.length();
        for (int i = 0; i < tLen; i++) {
            char c = t.charAt(i);
            ori.put(c, ori.getOrDefault(c, 0) + 1);
        }
        int l = 0, r = -1;
        int len = Integer.MAX_VALUE, ansL = -1, ansR = -1;
        int sLen = s.length();
        while (r < sLen) {
            ++r;
            if (r < sLen && ori.containsKey(s.charAt(r))) {
                cnt.put(s.charAt(r), cnt.getOrDefault(s.charAt(r), 0) + 1);
            }
            while (checkMinWindow() && l <= r) {
                if (r - l + 1 < len) {
                    len = r - l + 1;
                    ansL = l;
                    ansR = l + len;
                }
                if (ori.containsKey(s.charAt(l))) {
                    cnt.put(s.charAt(l), cnt.getOrDefault(s.charAt(l), 0) - 1);
                }
                ++l;
            }
        }
        return ansL == -1 ? "" : s.substring(ansL, ansR);
    }

    private boolean checkMinWindow(){
        Iterator iter = ori.entrySet().iterator();
        while (iter.hasNext()) {
            Map.Entry entry = (Map.Entry) iter.next();
            Character key = (Character) entry.getKey();
            Integer val = (Integer) entry.getValue();
            if (cnt.getOrDefault(key, 0) < val) {
                return false;
            }
        }
        return true;
    }

    /**
     * 394. 字符串解码
     *
     * 给定一个经过编码的字符串，返回它解码后的字符串。
     *
     * 编码规则为: k[encoded_string]，表示其中方括号内部的 encoded_string 正好重复 k 次。注意 k 保证为正整数。
     *
     * 你可以认为输入字符串总是有效的；输入字符串中没有额外的空格，且输入的方括号总是符合格式要求的。
     *
     * 此外，你可以认为原始数据不包含数字，所有的数字只表示重复的次数 k ，例如不会出现像 3a 或 2[4] 的输入。
     *
     * @param s
     * @return
     */
    int ptrDecodeString;
    public String decodeString(String s) {
        LinkedList<String> stk = new LinkedList<>();
        ptrDecodeString = 0;
        while (ptrDecodeString < s.length()){
            char cur = s.charAt(ptrDecodeString);
            if(Character.isDigit(cur)){
                String digits = getDigitsDecodeString(s);
                stk.addLast(digits);
            }else if(Character.isLetter(cur) || cur == '['){
                stk.addLast(String.valueOf(s.charAt(ptrDecodeString++)));
            }else{
                ++ptrDecodeString;
                LinkedList<String> sub = new LinkedList<>();
                while (!"[".equals(stk.peekLast())){
                    sub.addLast(stk.removeLast());
                }
                Collections.reverse(sub);
                stk.removeLast();
                int repTime = Integer.parseInt(stk.removeLast());
                StringBuffer t = new StringBuffer();
                String o = getStringDecodeString(sub);
                while (repTime-- > 0){
                    t.append(o);
                }
                stk.addLast(t.toString());
            }
        }
        return getStringDecodeString(stk);
    }

    private String getDigitsDecodeString(String s){
        StringBuffer ret = new StringBuffer();
        while (Character.isDigit(s.charAt(ptrDecodeString))){
            ret.append(s.charAt(ptrDecodeString++));
        }
        return ret.toString();
    }

    private String getStringDecodeString(LinkedList<String> v){
        StringBuffer ret = new StringBuffer();
        for(String s:v){
            ret.append(s);
        }
        return ret.toString();
    }
}
