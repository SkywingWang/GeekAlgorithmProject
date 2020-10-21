package algorithm;

import data.PalindromeNode;
import data.TreeNode;
import data.Trie;

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
        if (J == null || "".equals(J) || S == null || "".equals(S)){
            return 0;
        }
        char[] tmpJ = J.toCharArray();

        Set<Character> setJ = new HashSet<Character>();
        for (char c : tmpJ) {
            setJ.add(c);
        }
        int count = 0;
        char[] tmpS = S.toCharArray();
        for (int i = 0; i < tmpS.length; i++) {
            if (setJ.contains(tmpS[i])){
                count++;
            }
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

    /**
     * 126. 单词接龙 II
     * 给定两个单词（beginWord 和 endWord）和一个字典 wordList，找出所有从 beginWord 到 endWord 的最短转换序列。转换需遵循如下规则：
     *
     * 每次转换只能改变一个字母。
     * 转换过程中的中间单词必须是字典中的单词。
     * 说明:
     *
     * 如果不存在这样的转换序列，返回一个空列表。
     * 所有单词具有相同的长度。
     * 所有单词只由小写字母组成。
     * 字典中不存在重复的单词。
     * 你可以假设 beginWord 和 endWord 是非空的，且二者不相同。
     *
     * @param beginWord
     * @param endWord
     * @param wordList
     * @return
     */
    private static final int INF = 1 << 20;
    private Map<String,Integer> wordId = new HashMap<>();
    private ArrayList<String> idWord = new ArrayList<>();
    private ArrayList<Integer>[] edges;

    public List<List<String>> findLadders(String beginWord, String endWord, List<String> wordList) {
        int id = 0;
        for(String word : wordList){
            if(!wordId.containsKey(word)){
                wordId.put(word,id++);
                idWord.add(word);
            }
        }
        if(!wordId.containsKey(endWord)){
            return new ArrayList<>();
        }
        if(!wordId.containsKey(beginWord)){
            wordId.put(beginWord,id++);
            idWord.add(beginWord);
        }

        edges = new ArrayList[idWord.size()];
        for(int i = 0; i < idWord.size();i++){
            edges[i] = new ArrayList<>();
        }
        for(int i = 0; i < idWord.size();i++){
            for(int j = i + 1; j < idWord.size();j++){
                if(transformCheck(idWord.get(i),idWord.get(j))){
                    edges[i].add(j);
                    edges[j].add(i);
                }
            }
        }
        int dest = wordId.get(endWord);
        List<List<String>> res = new ArrayList<>();
        int[] cost = new int[id];
        for(int i = 0 ; i < id; i++){
            cost[i] = INF;
        }

        Queue<ArrayList<Integer>> q = new LinkedList<>();
        ArrayList<Integer> tmpBegin = new ArrayList<>();
        tmpBegin.add(wordId.get(beginWord));
        q.add(tmpBegin);
        cost[wordId.get(beginWord)] = 0;

        while (!q.isEmpty()){
            ArrayList<Integer> now = q.poll();
            int last = now.get(now.size() - 1);
            if(last == dest){
                ArrayList<String> tmp = new ArrayList<>();
                for(int index : now){
                    tmp.add(idWord.get(index));
                }
                res.add(tmp);
            } else{
                for(int i = 0 ; i < edges[last].size();i++){
                    int to = edges[last].get(i);
                    if(cost[last] + 1 <= cost[to]){
                        cost[to] = cost[last] + 1;
                        ArrayList<Integer> tmp = new ArrayList<>(now); tmp.add(to);
                        q.add(tmp);
                    }
                }
            }
        }
        return res;
    }

    private boolean transformCheck(String str1,String str2){
        int differences = 0;
        for(int i = 0; i < str1.length() && differences < 2; i++){
            if(str1.charAt(i) != str2.charAt(i)){
                ++differences;
            }
        }
        return differences == 1;
    }

    /**
     * 990. 等式方程的可满足性
     *
     * 给定一个由表示变量之间关系的字符串方程组成的数组，每个字符串方程 equations[i] 的长度为 4，并采用两种不同的形式之一："a==b" 或 "a!=b"。在这里，a 和 b 是小写字母（不一定不同），表示单字母变量名。
     *
     * 只有当可以将整数分配给变量名，以便满足所有给定的方程时才返回 true，否则返回 false。 
     *
     * @param equations
     * @return
     */
    public boolean equationsPossible(String[] equations) {
        if(equations == null || equations.length == 0)
            return false;
        int length = equations.length;
        int[] parent = new int[26];
        for(int i = 0; i < 26; i++){
            parent[i] = i;
        }
        for(String str : equations){
            if(str.charAt(1) == '='){
                int index1 = str.charAt(0) - 'a';
                int index2 = str.charAt(3) - 'a';
                union(parent,index1,index2);
            }
        }
        for(String str : equations){
            if(str.charAt(1) == '!'){
                int index1 = str.charAt(0) - 'a';
                int index2 = str.charAt(3) - 'a';
                if(find(parent,index1) == find(parent,index2)){
                    return false;
                }
            }
        }
        return true;
    }

    private void union(int[] parent, int index1, int index2){
        parent[find(parent,index1)] = find(parent,index2);
    }

    private int find(int[] parent,int index){
        while (parent[index] != index){
            parent[index] = parent[parent[index]];
            index = parent[index];
        }
        return index;
    }

    /**
     * 面试题46. 把数字翻译成字符串
     *
     * 给定一个数字，我们按照如下规则把它翻译为字符串：0 翻译成 “a” ，1 翻译成 “b”，……，11 翻译成 “l”，……，25 翻译成 “z”。一个数字可能有多个翻译。请编程实现一个函数，用来计算一个数字有多少种不同的翻译方法。
     *
     *
     * @param num
     * @return
     */
    public int translateNum(int num) {
        String src = String.valueOf(num);
        int p = 0, q = 0, r = 1;
        for(int i = 0; i < src.length(); i++){
            p = q;
            q = r;
            r = 0;
            r+= q;
            if(i== 0)
                continue;
            String pre = src.substring(i-1,i+1);
            if(pre.compareTo("25") <= 0 && pre.compareTo("10") >= 0)
                r += p;
        }
        return r;
    }

    /**
     * 10. 正则表达式匹配
     *
     * 给你一个字符串 s 和一个字符规律 p，请你来实现一个支持 '.' 和 '*' 的正则表达式匹配。
     *
     * '.' 匹配任意单个字符
     * '*' 匹配零个或多个前面的那一个元素
     * 所谓匹配，是要涵盖 整个 字符串 s的，而不是部分字符串。
     *
     * 说明:
     *
     * s 可能为空，且只包含从 a-z 的小写字母。
     * p 可能为空，且只包含从 a-z 的小写字母，以及字符 . 和 *。
     *
     *
     * @param s
     * @param p
     * @return
     */
    public boolean isMatch(String s, String p) {
        int m = s.length();
        int n = p.length();

        boolean[][] f = new boolean[m+1][n+1];
        f[0][0] = true;
        for(int i = 0; i <= m; i++){
            for(int j = 1; j <= n; j++){
                if(p.charAt(j - 1) == '*'){
                    f[i][j] = f[i][j-2];
                    if(matches(s,p,i,j-1)){
                        f[i][j] = f[i][j] || f[i-1][j];
                    }
                }else {
                    if(matches(s,p,i,j)){
                        f[i][j] = f[i - 1][j-1];
                    }
                }
            }
        }
        return f[m][n];
    }

    private boolean matches(String s,String p,int i,int j){
        if(i == 0)
            return false;
        if(p.charAt(j - 1) == '.')
            return true;
        return s.charAt(i - 1) == p.charAt(j - 1);
    }

    /**
     *
     * 面试题 16.18. 模式匹配
     *
     * 你有两个字符串，即pattern和value。 pattern字符串由字母"a"和"b"组成，用于描述字符串中的模式。
     * 例如，字符串"catcatgocatgo"匹配模式"aabab"（其中"cat"是"a"，"go"是"b"），
     * 该字符串也匹配像"a"、"ab"和"b"这样的模式。但需注意"a"和"b"不能同时表示相同的字符串。
     * 编写一个方法判断value字符串是否匹配pattern字符串。
     *
     * @param pattern
     * @param value
     * @return
     */
    public boolean patternMatching(String pattern, String value) {
        int count_a = 0, count_b = 0;
        for(char ch: pattern.toCharArray()){
            if(ch == 'a'){
                ++count_a;
            }else{
                ++count_b;
            }
        }
        if(count_a < count_b){
            int temp = count_a;
            count_a = count_b;
            count_b = temp;
            char[] array = pattern.toCharArray();
            for(int i = 0; i < array.length; i++){
                array[i] = array[i] == 'a' ? 'b' : 'a';
            }
            pattern = new String(array);
        }
        if(value.length() == 0){
            return count_b == 0;
        }
        if(pattern.length() == 0){
            return false;
        }
        for(int len_a = 0; count_a * len_a <= value.length(); ++len_a){
            int rest = value.length() -  count_a * len_a;
            if((count_b == 0 && rest == 0)|| (count_b != 0 && rest % count_b == 0)){
                int len_b = (count_b == 0 ? 0:rest/count_b);
                int pos = 0;
                boolean correct = true;
                String value_a = "",value_b = "";
                for(char ch : pattern.toCharArray()){
                    if(ch == 'a'){
                        String sub = value.substring(pos,pos + len_a);
                        if(value_a.length() == 0){
                            value_a = sub;
                        }else if(!value_a.equals(sub)){
                            correct = false;
                            break;
                        }
                        pos += len_a;
                    }else {
                        String sub = value.substring(pos,pos + len_b);
                        if(value_b.length() == 0){
                            value_b = sub;
                        }else if(!value_b.equals(sub)){
                            correct = false;
                            break;
                        }
                        pos += len_b;
                    }
                }
                if(correct && !value_a.equals(value_b)){
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * 67. 二进制求和
     *
     * 给你两个二进制字符串，返回它们的和（用二进制表示）。
     *
     * 输入为 非空 字符串且只包含数字 1 和 0。
     *
     * @param a
     * @param b
     * @return
     */
    public String addBinary_V2(String a, String b) {
        if(a == null || "".trim().equals(a))
            return b;
        if(b == null || "".trim().equals(b))
            return a;
        int n = Math.max(a.length(),b.length()),carry = 0;
        StringBuffer resultSB = new StringBuffer();
        for(int i = 0; i < n; i++){
            carry += i < a.length() ? (a.charAt(a.length() - 1 -i) - '0') : 0;
            carry += i < b.length() ? (b.charAt(b.length() - 1 - i) - '0') : 0;
            resultSB.append((char)(carry % 2 + '0'));
            carry /= 2;
        }
        if(carry > 0)
            resultSB.append('1');
        resultSB.reverse();
        return resultSB.toString();
    }

    /**
     * 139. 单词拆分
     *
     * 给定一个非空字符串 s 和一个包含非空单词列表的字典 wordDict，判定 s 是否可以被空格拆分为一个或多个在字典中出现的单词。
     *
     * 说明：
     *
     * 拆分时可以重复使用字典中的单词。
     * 你可以假设字典中没有重复的单词。
     *
     *
     * @param s
     * @param wordDict
     * @return
     */
    public boolean wordBreak(String s, List<String> wordDict) {
        if(s==null || "".equals(s))
            return true;
        if(wordDict == null || wordDict.size() == 0)
            return false;
        Set<String> wordDictSet = new HashSet(wordDict);
        boolean[] dp = new boolean[s.length() + 1];
        dp[0] = true;
        for (int i = 1; i <= s.length(); i++) {
            for (int j = 0; j < i; j++) {
                if (dp[j] && wordDictSet.contains(s.substring(j, i))) {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[s.length()];
    }

    /**
     * 44. 通配符匹配
     *
     *  给定一个字符串 (s) 和一个字符模式 (p) ，实现一个支持 '?' 和 '*' 的通配符匹配。
     *
     * '?' 可以匹配任何单个字符。
     * '*' 可以匹配任意字符串（包括空字符串）。
     * 两个字符串完全匹配才算匹配成功。
     *
     * @param s
     * @param p
     * @return
     */
    public boolean isMatchC(String s, String p) {
        int m = s.length();
        int n = p.length();
        boolean[][] dp = new boolean[m+1][n+1];
        dp[0][0] = true;
        for(int i = 1; i <= n; i++){
            if(p.charAt(i - 1) == '*'){
                dp[0][i] = true;

            }else
                break;
        }
        for(int i = 1; i <= m; i++){
            for(int j = 1; j <= n; j++){
                if(p.charAt(j - 1)=='*'){
                    dp[i][j] = dp[i][j - 1] || dp[i-1][j];
                }else if(p.charAt(j - 1) == '?' || s.charAt(i - 1) == p.charAt(j - 1)){
                    dp[i][j] = dp[i - 1][j - 1];
                }
            }
        }
        return dp[m][n];
    }

    /**
     * 面试题 17.13. 恢复空格
     * 哦，不！你不小心把一个长篇文章中的空格、标点都删掉了，并且大写也弄成了小写。
     * 像句子"I reset the computer. It still didn’t boot!"已经变成了"iresetthecomputeritstilldidntboot"。
     * 在处理标点符号和大小写之前，你得先把它断成词语。当然了，你有一本厚厚的词典dictionary，不过，有些词没在词典里。
     * 假设文章用sentence表示，设计一个算法，把文章断开，要求未识别的字符最少，返回未识别的字符数。
     *
     * @param dictionary
     * @param sentence
     * @return
     */
    public int respace(String[] dictionary, String sentence) {
        int n = sentence.length();
        Trie root = new Trie();
        for(String word : dictionary){
            root.insert(word);
        }
        int[] dp = new int[n + 1];
        Arrays.fill(dp,Integer.MAX_VALUE);
        dp[0] = 0;
        for(int i = 1; i <= n; i++){
            dp[i] = dp[i-1] + 1;
            Trie curPos = root;
            for(int j = i; j >= 1;j--){
                int t = sentence.charAt(j - 1) - 'a';
                if(curPos.next[t] == null)
                    break;
                else if(curPos.next[t].isEnd){
                    dp[i] = Math.min(dp[i],dp[j-1]);
                }
                if(dp[i]==0)
                    break;
                curPos = curPos.next[t];
            }
        }
        return dp[n];
    }

    /**
     *
     * 97. 交错字符串
     *
     * 给定三个字符串 s1, s2, s3, 验证 s3 是否是由 s1 和 s2 交错组成的。
     *
     * @param s1
     * @param s2
     * @param s3
     * @return
     */
    public boolean isInterleave(String s1, String s2, String s3) {
        if(s1 == null || s2 == null || s3 == null || s1.length() == 0 || s2.length() == 0 || s3.length() == 0 || s3.length() != s1.length() + s2.length())
            return false;
        int s1Index = 0,s2Index = 0, s3Index = 0;
        int s1Length = s1.length(),s2Length = s2.length(),s3Length = s3.length();

        boolean[][] f = new boolean[s1Length + 1][s2Length + 1];
        f[0][0] = true;
        for(int i = 0; i <= s1Length; i++){
            for(int j = 0; j <= s2Length; j++){
                int p = i + j - 1;
                if(i > 0)
                    f[i][j] = f[i][j] || (f[i-1][j] && s1.charAt(i - 1) == s3.charAt(p));
                if(j > 0)
                    f[i][j] = f[i][j] || (f[i][j-1] && s2.charAt(j-1) == s3.charAt(p));
            }
        }
        return f[s1Length][s2Length];
    }

    /**
     *
     * 312. 戳气球
     *
     * 有 n 个气球，编号为0 到 n-1，每个气球上都标有一个数字，这些数字存在数组 nums 中。
     *
     * 现在要求你戳破所有的气球。如果你戳破气球 i ，就可以获得 nums[left] * nums[i] * nums[right] 个硬币。 这里的 left 和 right 代表和 i 相邻的两个气球的序号。注意当你戳破了气球 i 后，气球 left 和气球 right 就变成了相邻的气球。
     *
     * 求所能获得硬币的最大数量。
     *
     *
     * @param nums
     * @return
     */
    public int[][] rec;
    public int[] val;
    public int maxCoins(int[] nums) {
        int n = nums.length;
        val = new int[n + 2];
        for(int i = 1; i <= n ; i++){
            val[i] = nums[i-1];
        }
        val[0] = val[n + 1] = 1;
        rec = new int[n+2][n+2];
        for(int i = 0; i <= n + 1; i++){
            Arrays.fill(rec[i],-1);
        }
        return solveMaxConins(0,n + 1);
    }

    private int solveMaxConins(int left,int right){
        if(left >= right - 1)
            return 0;
        if(rec[left][right] != -1)
            return rec[left][right];
        for(int i = left + 1; i < right; i++){
            int sum = val[left] * val[i] * val[right];
            sum += solveMaxConins(left,i) + solveMaxConins(i,right);
            rec[left][right] = Math.max(rec[left][right],sum);
        }
        return rec[left][right];
    }

    /**
     * 1486. 数组异或操作
     *
     * 给你两个整数，n 和 start 。
     *
     * 数组 nums 定义为：nums[i] = start + 2*i（下标从 0 开始）且 n == nums.length 。
     *
     * 请返回 nums 中所有元素按位异或（XOR）后得到的结果。
     *
     *
     * @param n
     * @param start
     * @return
     */
    public int xorOperation(int n, int start) {
        int result = start;
        for(int i = 1; i < n ; i ++){
            start += 2;
            result ^= start;
        }
        return result;
    }

    /**
     * 95. 不同的二叉搜索树 II
     *
     * 给定一个整数 n，生成所有由 1 ... n 为节点所组成的 二叉搜索树 。
     *
     * @param n
     * @return
     */
    public List<TreeNode> generateTrees(int n) {
        if( n == 0)
            return new LinkedList<TreeNode>();
        return generateTrees(1,n);
    }

    private List<TreeNode> generateTrees(int start,int end){
        List<TreeNode> allTrees = new LinkedList<TreeNode>();
        if(start > end){
            allTrees.add(null);
            return allTrees;
        }

        for(int i = start; i <= end; i++){
            List<TreeNode> leftTrees = generateTrees(start,i - 1);
            List<TreeNode> rightTrees = generateTrees(i + 1,end);
            for(TreeNode left:leftTrees){
                for(TreeNode right : rightTrees){
                    TreeNode currTree = new TreeNode(i);
                    currTree.left = left;
                    currTree.right = right;
                    allTrees.add(currTree);
                }
            }

        }
        return allTrees;
    }

    /**
     * 392. 判断子序列
     *
     * 给定字符串 s 和 t ，判断 s 是否为 t 的子序列。
     *
     * 你可以认为 s 和 t 中仅包含英文小写字母。字符串 t 可能会很长（长度 ~= 500,000），而 s 是个短字符串（长度 <=100）。
     *
     * 字符串的一个子序列是原始字符串删除一些（也可以不删除）字符而不改变剩余字符相对位置形成的新字符串。（例如，"ace"是"abcde"的一个子序列，而"aec"不是）。
     *
     *
     * @param s
     * @param t
     * @return
     */
    public boolean isSubsequence(String s, String t) {
        if(s == null  || t == null)
            return false;
        int sIndex = 0, tIndex = 0;
        while (sIndex < s.length() && tIndex < t.length()){
            if(s.charAt(sIndex) == t.charAt(tIndex)){
                sIndex++;
            }
            tIndex++;
        }
        return sIndex == s.length();
    }

    public void testStr(String fileStr){
        String[] str = fileStr.split("\\.");
        System.out.println(str[fileStr.split("\\.").length - 1]);
    }

    /**
     * 415. 字符串相加
     * 给定两个字符串形式的非负整数 num1 和num2 ，计算它们的和。
     *
     * 注意：
     *
     * num1 和num2 的长度都小于 5100.
     * num1 和num2 都只包含数字 0-9.
     * num1 和num2 都不包含任何前导零。
     * 你不能使用任何內建 BigInteger 库， 也不能直接将输入的字符串转换为整数形式。
     *
     * @param num1
     * @param num2
     * @return
     */
    public String addStrings(String num1, String num2) {
        if(num1 == null && num2 == null)
            return null;
        if(num1 == null)
            return num2;
        if(num2 == null)
            return num1;
        StringBuffer result = new StringBuffer();
        int i = num1.length() - 1;
        int j = num2.length() - 1;
        int add = 0;
        while (i >= 0 || j >= 0 || add != 0){
            int x = i >= 0?num1.charAt(i) - '0' : 0;
            int y = j >= 0?num2.charAt(j) - '0' : 0;
            int r = x + y + add;
            result.append(r % 10);
            add = r / 10;
            i--;
            j--;

        }
        result.reverse();
        return result.toString();
    }

    /**
     * 336. 回文对
     *
     * 给定一组 互不相同 的单词， 找出所有不同 的索引对(i, j)，使得列表中的两个单词， words[i] + words[j] ，可拼接成回文串。
     *
     * @param words
     * @return
     */
    List<PalindromeNode> palindromeNodeTree = new ArrayList<>();
    public List<List<Integer>> palindromePairs(String[] words) {
        if(words == null || words.length == 0)
            return null;
        palindromeNodeTree.add(new PalindromeNode());
        int n = words.length;
        for(int i = 0; i < n;i++)
            insertPalindromePairs(words[i],i);
        List<List<Integer>> ret = new ArrayList<>();
        for(int i = 0; i < n; i++){
            int m = words[i].length();
            for(int j = 0; j <= m; j++){
                if(isPalindromePairs(words[i],j,m-1)){
                    int leftId = findWordPalindromePairs(words[i],0,j - 1);
                    if(leftId != -1 && leftId != i){
                        ret.add(Arrays.asList(i,leftId));
                    }
                }
                if(j != 0 && isPalindromePairs(words[i],0,j-1)){
                    int rightId = findWordPalindromePairs(words[i],j,m-1);
                    if(rightId != -1 && rightId != i){
                        ret.add(Arrays.asList(rightId,i));
                    }
                }
            }
        }
        return ret;
    }

    private void insertPalindromePairs(String s,int id){
        int len = s.length(), add = 0;
        for(int i = 0; i < len; i++){
            int x = s.charAt(i) - 'a';
            if(palindromeNodeTree.get(add).ch[x] == 0){
                palindromeNodeTree.add(new PalindromeNode());
                palindromeNodeTree.get(add).ch[x] = palindromeNodeTree.size() - 1;
            }
            add = palindromeNodeTree.get(add).ch[x];
        }
        palindromeNodeTree.get(add).flag = id;
    }

    private boolean isPalindromePairs(String s,int left,int right){
        int len = right - left + 1;
        for(int i = 0; i < len / 2; i++){
            if(s.charAt(left + i) != s.charAt(right - i)){
                return false;
            }
        }
        return true;
    }

    public int findWordPalindromePairs(String s,int left, int right){
        int add = 0;
        for(int i = right; i >= left; i--){
            int x = s.charAt(i) - 'a';
            if(palindromeNodeTree.get(add).ch[x] == 0)
                return -1;
            add = palindromeNodeTree.get(add).ch[x];
        }
        return palindromeNodeTree.get(add).flag;
    }

    /**
     * 93. 复原IP地址
     *
     * 给定一个只包含数字的字符串，复原它并返回所有可能的 IP 地址格式。
     *
     * 有效的 IP 地址正好由四个整数（每个整数位于 0 到 255 之间组成），整数之间用 '.' 分隔。
     *
     * @param s
     * @return
     */
    static final int SEG_COUNT = 4;
    List<String> ansRestoreIpAddresses = new ArrayList<>();
    int[] segments = new int[SEG_COUNT];
    public List<String> restoreIpAddresses(String s) {
        segments = new int[SEG_COUNT];
        dfsRestoreIpAddress(s,0,0);
        return ansRestoreIpAddresses;
    }

    private void dfsRestoreIpAddress(String s,int segId,int segStart){
        if(segId == SEG_COUNT){
            if(segStart == s.length()){
                StringBuffer ipAddr = new StringBuffer();
                for(int i = 0; i < SEG_COUNT; i++){
                    ipAddr.append(segments[i]);
                    if(i != SEG_COUNT - 1)
                        ipAddr.append('.');
                }
                ansRestoreIpAddresses.add(ipAddr.toString());
            }
            return;
        }
        if(segStart == s.length())
            return;

        if(s.charAt(segStart) == '0'){
            segments[segId] = 0;
            dfsRestoreIpAddress(s,segId + 1,segStart + 1);
        }
        int addr = 0;
        for(int segEnd = segStart; segEnd < s.length(); segEnd++){
            addr = addr * 10 + (s.charAt(segEnd) - '0');
            if(addr > 0 && addr <= 0xFF){
                segments[segId] = addr;
                dfsRestoreIpAddress(s,segId + 1, segEnd + 1);
            }else
                break;
        }
    }

    /**
     * 696. 计数二进制子串
     *
     * 给定一个字符串 s，计算具有相同数量0和1的非空(连续)子字符串的数量，并且这些子字符串中的所有0和所有1都是组合在一起的。
     *
     * 重复出现的子串要计算它们出现的次数。
     *
     *
     * @param s
     * @return
     */
    public int countBinarySubstrings(String s) {
        if(s == null || "".equals(s))
            return 0;
        int ptr = 0, n = s.length(), last = 0,ans = 0;
        while (ptr < n){
            char c = s.charAt(ptr);
            int count = 0;
            while (ptr < n && s.charAt(ptr) == c){
                ptr++;
                count++;
            }
            ans += Math.min(count,last);
            last = count;
        }
        return ans;
    }

    /**
     * 43. 字符串相乘
     *
     * 给定两个以字符串形式表示的非负整数 num1 和 num2，返回 num1 和 num2 的乘积，它们的乘积也表示为字符串形式。
     *
     * @param num1
     * @param num2
     * @return
     */
    public String multiply(String num1, String num2) {
        if(num1.equals("0") || num2.equals("0")){
            return "0";
        }
        int m = num1.length(), n = num2.length();
        int[] ansArr = new int[m + n];
        for(int i = m - 1; i >= 0; i--){
            int x = num1.charAt(i) - '0';
            for(int j = n - 1; j >= 0; j--){
                int y = num2.charAt(j) - '0';
                ansArr[i + j + 1] += x * y;
            }
        }
        for(int i = m + n - 1; i > 0; i--){
            ansArr[i - 1] += ansArr[i] / 10;
            ansArr[i] %= 10;
        }
        int index = ansArr[0] == 0 ? 1 : 0;
        StringBuffer ans = new StringBuffer();
        while (index < m + n){
            ans.append(ansArr[index]);
            index++;
        }
        return ans.toString();
    }

    /**
     * 647. 回文子串
     *
     * 给定一个字符串，你的任务是计算这个字符串中有多少个回文子串。
     *
     * 具有不同开始位置或结束位置的子串，即使是由相同的字符组成，也会被视作不同的子串。
     *
     *
     * @param s
     * @return
     */
    public int countSubstrings(String s) {
        int n = s.length(), ans = 0;
        for(int i = 0; i < 2 * n - 1; i++){
            int l = i / 2, r= i / 2 + i % 2;
            while (l >= 0 && r < n && s.charAt(l) == s.charAt(r)){
                l--;
                r++;
                ans++;
            }
        }
        return ans;
    }

    /**
     *
     * @param s
     * @return
     */
    public boolean repeatedSubstringPattern(String s) {
        return kmpRepeatedSubstringPattern(s + s,s);
    }

    public boolean kmpRepeatedSubstringPattern(String query,String pattern){
        int n = query.length();
        int m = pattern.length();
        int[] fail = new int[m];
        Arrays.fill(fail, -1);
        for(int i = 1; i < m; i++){
            int j = fail[i - 1];
            while (j != -1 && pattern.charAt(j + 1) != pattern.charAt(i)){
                j = fail[j];
            }
            if(pattern.charAt(j+1)==pattern.charAt(i)){
                fail[i] = j+ 1;
            }
        }
        int match = -1;
        for(int i = 1; i < n - 1; i++){
            while (match != -1 && pattern.charAt(match + 1) != query.charAt(i)){
                match = fail[match];
            }
            if(pattern.charAt(match + 1) == query.charAt(i)){
                match++;
                if(match == m - 1){
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * 17. 电话号码的字母组合
     *
     * 给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。
     *
     * 给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。
     *
     * @param digits
     * @return
     */
    public List<String> letterCombinations(String digits) {
        List<String> combinations = new ArrayList<>();
        if(digits.length() == 0){
            return combinations;
        }
        Map<Character,String> phoneMap = new HashMap<Character, String>(){{
            put('2', "abc");
            put('3', "def");
            put('4', "ghi");
            put('5', "jkl");
            put('6', "mno");
            put('7', "pqrs");
            put('8', "tuv");
            put('9', "wxyz");
        }};
        backtrackLetterCombinations(combinations,phoneMap,digits,0,new StringBuffer());
        return combinations;
    }

    private void backtrackLetterCombinations(List<String> combinations, Map<Character,String> phoneMap, String digits,int index,StringBuffer combination){
        if(index == digits.length()){
            combinations.add(combination.toString());
        }else{
            char digit = digits.charAt(index);
            String letters = phoneMap.get(digit);
            int lettersCount = letters.length();
            for(int i = 0; i < lettersCount; i++){
                combination.append(letters.charAt(i));
                backtrackLetterCombinations(combinations,phoneMap,digits,index + 1,combination);
                combination.deleteCharAt(index);
            }
        }
    }

    /**
     * 657. 机器人能否返回原点
     *
     * 在二维平面上，有一个机器人从原点 (0, 0) 开始。给出它的移动顺序，判断这个机器人在完成移动后是否在 (0, 0) 处结束。
     *
     * 移动顺序由字符串表示。字符 move[i] 表示其第 i 次移动。机器人的有效动作有 R（右），L（左），U（上）和 D（下）。如果机器人在完成所有动作后返回原点，则返回 true。否则，返回 false。
     *
     * 注意：机器人“面朝”的方向无关紧要。 “R” 将始终使机器人向右移动一次，“L” 将始终向左移动等。此外，假设每次移动机器人的移动幅度相同。
     *
     *
     * @param moves
     * @return
     */
    public boolean judgeCircle(String moves) {
        if(moves == null || moves.length() == 0)
            return true;
        int lr = 0, ud = 0;
        for(int i = 0 ; i < moves.length();i ++){
            char c = moves.charAt(i);
            if(c == 'L'){
                lr++;
            }else if(c == 'R'){
                lr--;
            }else if(c == 'U'){
                ud++;
            }else if(c == 'D'){
                ud--;
            }
        }
        if(lr == 0 && ud == 0){
            return true;
        }
        return false;
    }

    /**
     * 557. 反转字符串中的单词 III
     * 给定一个字符串，你需要反转字符串中每个单词的字符顺序，同时仍保留空格和单词的初始顺序。
     * @param s
     * @return
     */
    public String reverseWords_3(String s) {
        StringBuffer ret = new StringBuffer();
        int length = s.length();
        int i = 0;
        while (i < length) {
            int start = i;
            while (i < length && s.charAt(i) != ' ') {
                i++;
            }
            for (int p = start; p < i; p++) {
                ret.append(s.charAt(start + i - 1 - p));
            }
            while (i < length && s.charAt(i) == ' ') {
                i++;
                ret.append(' ');
            }
        }
        return ret.toString();

    }

    /**
     * 剑指 Offer 20. 表示数值的字符串
     *
     * 请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。例如，字符串"+100"、"5e2"、"-123"、"3.1416"、"-1E-16"、"0123"都表示数值，但"12e"、"1a3.14"、"1.2.3"、"+-5"及"12e+5.4"都不是。
     *
     * @param s
     * @return
     */
    public boolean isNumber(String s) {
        Map<State, Map<CharType, State>> transfer = new HashMap<State, Map<CharType, State>>();
        Map<CharType, State> initialMap = new HashMap<CharType, State>() {{
            put(CharType.CHAR_SPACE, State.STATE_INITIAL);
            put(CharType.CHAR_NUMBER, State.STATE_INTEGER);
            put(CharType.CHAR_POINT, State.STATE_POINT_WITHOUT_INT);
            put(CharType.CHAR_SIGN, State.STATE_INT_SIGN);
        }};
        transfer.put(State.STATE_INITIAL, initialMap);
        Map<CharType, State> intSignMap = new HashMap<CharType, State>() {{
            put(CharType.CHAR_NUMBER, State.STATE_INTEGER);
            put(CharType.CHAR_POINT, State.STATE_POINT_WITHOUT_INT);
        }};
        transfer.put(State.STATE_INT_SIGN, intSignMap);
        Map<CharType, State> integerMap = new HashMap<CharType, State>() {{
            put(CharType.CHAR_NUMBER, State.STATE_INTEGER);
            put(CharType.CHAR_EXP, State.STATE_EXP);
            put(CharType.CHAR_POINT, State.STATE_POINT);
            put(CharType.CHAR_SPACE, State.STATE_END);
        }};
        transfer.put(State.STATE_INTEGER, integerMap);
        Map<CharType, State> pointMap = new HashMap<CharType, State>() {{
            put(CharType.CHAR_NUMBER, State.STATE_FRACTION);
            put(CharType.CHAR_EXP, State.STATE_EXP);
            put(CharType.CHAR_SPACE, State.STATE_END);
        }};
        transfer.put(State.STATE_POINT, pointMap);
        Map<CharType, State> pointWithoutIntMap = new HashMap<CharType, State>() {{
            put(CharType.CHAR_NUMBER, State.STATE_FRACTION);
        }};
        transfer.put(State.STATE_POINT_WITHOUT_INT, pointWithoutIntMap);
        Map<CharType, State> fractionMap = new HashMap<CharType, State>() {{
            put(CharType.CHAR_NUMBER, State.STATE_FRACTION);
            put(CharType.CHAR_EXP, State.STATE_EXP);
            put(CharType.CHAR_SPACE, State.STATE_END);
        }};
        transfer.put(State.STATE_FRACTION, fractionMap);
        Map<CharType, State> expMap = new HashMap<CharType, State>() {{
            put(CharType.CHAR_NUMBER, State.STATE_EXP_NUMBER);
            put(CharType.CHAR_SIGN, State.STATE_EXP_SIGN);
        }};
        transfer.put(State.STATE_EXP, expMap);
        Map<CharType, State> expSignMap = new HashMap<CharType, State>() {{
            put(CharType.CHAR_NUMBER, State.STATE_EXP_NUMBER);
        }};
        transfer.put(State.STATE_EXP_SIGN, expSignMap);
        Map<CharType, State> expNumberMap = new HashMap<CharType, State>() {{
            put(CharType.CHAR_NUMBER, State.STATE_EXP_NUMBER);
            put(CharType.CHAR_SPACE, State.STATE_END);
        }};
        transfer.put(State.STATE_EXP_NUMBER, expNumberMap);
        Map<CharType, State> endMap = new HashMap<CharType, State>() {{
            put(CharType.CHAR_SPACE, State.STATE_END);
        }};
        transfer.put(State.STATE_END, endMap);

        int length = s.length();
        State state = State.STATE_INITIAL;

        for (int i = 0; i < length; i++) {
            CharType type = toCharType(s.charAt(i));
            if (!transfer.get(state).containsKey(type)) {
                return false;
            } else {
                state = transfer.get(state).get(type);
            }
        }
        return state == State.STATE_INTEGER || state == State.STATE_POINT || state == State.STATE_FRACTION || state == State.STATE_EXP_NUMBER || state == State.STATE_END;
    }

    public CharType toCharType(char ch){
        if (ch >= '0' && ch <= '9') {
            return CharType.CHAR_NUMBER;
        } else if (ch == 'e' || ch == 'E') {
            return CharType.CHAR_EXP;
        } else if (ch == '.') {
            return CharType.CHAR_POINT;
        } else if (ch == '+' || ch == '-') {
            return CharType.CHAR_SIGN;
        } else if (ch == ' ') {
            return CharType.CHAR_SPACE;
        } else {
            return CharType.CHAR_ILLEGAL;
        }
    }

    enum State {
        STATE_INITIAL,
        STATE_INT_SIGN,
        STATE_INTEGER,
        STATE_POINT,
        STATE_POINT_WITHOUT_INT,
        STATE_FRACTION,
        STATE_EXP,
        STATE_EXP_SIGN,
        STATE_EXP_NUMBER,
        STATE_END,
    }

    enum CharType {
        CHAR_NUMBER,
        CHAR_EXP,
        CHAR_POINT,
        CHAR_SIGN,
        CHAR_SPACE,
        CHAR_ILLEGAL,
    }

    /**
     * 344. 反转字符串
     * 编写一个函数，其作用是将输入的字符串反转过来。输入字符串以字符数组 char[] 的形式给出。
     *
     * 不要给另外的数组分配额外的空间，你必须原地修改输入数组、使用 O(1) 的额外空间解决这一问题。
     *
     * 你可以假设数组中的所有字符都是 ASCII 码表中的可打印字符。
     *
     * @param s
     */
    public void reverseString(char[] s) {
        if(s == null){
            return;}
        int n = s.length;
        for(int left = 0,right = n - 1; left < right; left++,right--){
            char tmp = s[left];
            s[left] = s[right];
            s[right] = tmp;
        }
    }

    /**
     * 1002. 查找常用字符
     * 给定仅有小写字母组成的字符串数组 A，返回列表中的每个字符串中都显示的全部字符（包括重复字符）组成的列表。例如，如果一个字符在每个字符串中出现 3 次，但不是 4 次，则需要在最终答案中包含该字符 3 次。
     *
     * 你可以按任意顺序返回答案。
     *
     * @param A
     * @return
     */
    public List<String> commonChars(String[] A) {
        if(A == null){
            return null;
        }
        int[] minfreq = new int[26];
        Arrays.fill(minfreq,Integer.MAX_VALUE);
        for(String word:A){
            int[] freq = new int[26];
            int length = word.length();
            for(int i = 0; i < length; i++){
                char ch = word.charAt(i);
                ++freq[ch - 'a'];
            }
            for(int i = 0;i < 26;i++){
                minfreq[i] = Math.min(minfreq[i],freq[i]);
            }
        }
        List<String> ans = new ArrayList<>();
        for(int i = 0; i < 26; i++){
            for(int j = 0; j < minfreq[i];j++){
                ans.add(String.valueOf((char)(i + 'a')));
            }
        }
        return ans;
    }

    /**
     * 844. 比较含退格的字符串
     *
     * 给定 S 和 T 两个字符串，当它们分别被输入到空白的文本编辑器后，判断二者是否相等，并返回结果。 # 代表退格字符。
     *
     * 注意：如果对空文本输入退格字符，文本继续为空。
     *
     * @param S
     * @param T
     * @return
     */
    public boolean backspaceCompare(String S, String T) {
        int i = S.length() - 1,j = T.length() - 1;
        int skipS = 0, skipT = 0;
        while (i >= 0 || j >= 0){
            while (i >= 0){
                if(S.charAt(i) == '#'){
                    skipS++;
                    i--;
                }else if(skipS > 0){
                    skipS--;
                    i--;
                }else{
                    break;
                }
            }
            while (j >= 0){
                if(T.charAt(j) == '#'){
                    skipT++;
                    j--;
                }else if(skipT > 0){
                    skipT--;
                    j--;
                }else{
                    break;
                }
            }
            if(i >= 0 && j >= 0){
                if(S.charAt(i) != T.charAt(j)){
                    return false;
                }
            }else{
                if(i >= 0 || j >=0){
                    return false;
                }
            }
            i--;
            j--;
        }
        return true;
    }

    /**
     * 925. 长按键入
     *
     * 你的朋友正在使用键盘输入他的名字 name。偶尔，在键入字符 c 时，按键可能会被长按，而字符可能被输入 1 次或多次。
     *
     * 你将会检查键盘输入的字符 typed。如果它对应的可能是你的朋友的名字（其中一些字符可能被长按），那么就返回 True。
     *
     * @param name
     * @param typed
     * @return
     */
    public boolean isLongPressedName(String name, String typed) {
        int i = 0, j = 0;
        while (j < typed.length()){
            if(i < name.length() && name.charAt(i) == typed.charAt(j)){
                i++;
                j++;
            }else if(j > 0 && typed.charAt(j) == typed.charAt(j - 1)){
                j++;
            }else{
                return false;
            }
        }
        return i == name.length();
    }
}
