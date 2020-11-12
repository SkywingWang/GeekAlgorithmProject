package algorithm;

import data.MountainArray;
import data.TreeNode;

import java.util.*;

/**
 * created by Sven
 * on 2019-12-14
 * <p>
 * 数组Array的相关算法
 */
public class ArrayAlgorithm {

    /**
     * created by Sven
     * on 2019-12-07
     * <p>
     * 加一
     * <p>
     * 给定一个由整数组成的非空数组所表示的非负整数，在该数的基础上加一。
     * <p>
     * 最高位数字存放在数组的首位， 数组中每个元素只存储单个数字。
     * <p>
     * 你可以假设除了整数 0 之外，这个整数不会以零开头。
     */
    public int[] plusOne(int[] digits) {
        if (digits == null)
            return digits;
        int i = digits.length - 1;
        boolean isCarry = false;
        while (i > 0) {
            if (isCarry) {
                if (digits[i] + 1 == 10) {
                    isCarry = true;
                    digits[i] = 0;
                } else {
                    digits[i] = digits[i] + 1;
                    isCarry = false;
                    break;
                }
            } else {
                if (digits[i] + 1 == 10) {
                    isCarry = true;
                    digits[i] = 0;
                } else {
                    digits[i] = digits[i] + 1;
                    isCarry = false;
                    break;
                }
            }
            i--;
        }
        if (isCarry || digits.length == 1) {
            if (digits[0] + 1 == 10) {
                digits = new int[digits.length + 1];
                digits[0] = 1;
            } else {
                digits[0] = digits[0] + 1;
            }
        }
        return digits;
    }

    /**
     * created by Sven
     * on 2019-12-09
     * <p>
     * 数组中的连续子数组的最大和
     */
    public int maxSubArray(int[] nums) {
        int ans = nums[0];
        int sum = 0;
        for (int num : nums) {
            if (sum > 0) {
                sum += num;
            } else {
                sum = num;
            }
            ans = Math.max(ans, sum);
        }
        return ans;
    }

    /**
     * created by Sven
     * on 2019-12-02
     * <p>
     * 合并两个有序数组
     */
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        if (nums1 == null || nums2 == null || nums2.length == 0)
            return;
        int i = m, j = n;
        while (i > 0 && j > 0) {
            if (nums1[i - 1] > nums2[j - 1]) {
                nums1[i + j - 1] = nums1[i - 1];
                i--;
            } else {
                nums1[i + j - 1] = nums2[j - 1];
                j--;
            }
        }
        while (i > 0) {
            nums1[i + j - 1] = nums1[i - 1];
            i--;
        }
        while (j > 0) {
            nums1[i + j - 1] = nums2[j - 1];
            j--;
        }

    }

    public void merge1(int[] nums1, int m, int[] nums2, int n) {
        if (nums1 == null || nums2 == null || nums2.length == 0)
            return;
        int tmp = 0;
        for (int i = 0; i < m; i++) {
            if (nums1[i] > nums2[0]) {
                tmp = nums1[i];
                nums1[i] = nums2[0];
                boolean isChange = false;
                for (int j = 0; j < n - 1; j++) {
                    if (nums2[j + 1] < tmp) {
                        nums2[j] = nums2[j + 1];
                    } else {
                        nums2[j] = tmp;
                        isChange = true;
                        break;
                    }
                }
                if (!isChange)
                    nums2[n - 1] = tmp;
            }
        }
        for (int i = 0; i < n; i++) {
            nums1[m + i] = nums2[i];
        }
    }

    /**
     * created by Sven
     * on 2019-11-29
     * <p>
     * 删除数组重复元素
     */
    public int removeDuplicates(int[] nums) {
        if (nums == null)
            return 0;
        if (nums.length < 2)
            return nums.length;
        int k = 0;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] != nums[k]) {
                k++;
                nums[k] = nums[i];
            }
        }
        return k + 1;

    }

    /**
     * created by Sven
     * on 2019-11-29
     * <p>
     * 数组右移N步
     */
    public void rotate(int[] nums, int k) {
        if (nums == null || nums.length == 0 || k == 0)
            return;
        k = k % nums.length;
        reverse(nums, 0, nums.length - 1);
        reverse(nums, 0, k - 1);
        reverse(nums, k, nums.length - 1);
        return;
    }

    private void reverse(int[] nums, int left, int right) {
        while (left < right)
            swap(nums, left++, right--);

    }

    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    public void rotateMy(int[] nums, int k) {
        if (nums == null || nums.length == 0 || k == 0)
            return;
        k = k % nums.length;
        for (int i = 0; i < nums.length / 2; i++) {
            swap(nums, i, nums.length - 1 - i);
        }
        for (int i = 0; i < k / 2; i++) {
            swap(nums, i, k - 1 - i);
        }
        for (int i = k; i < (nums.length + k) / 2; i++) {
            swap(nums, i, nums.length + k - i - 1);
        }
    }

    /**
     * created by Sven
     * on 2019-12-02
     * <p>
     * 搜索插入位置
     */
    public int searchInsert(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            nums = new int[]{target};
            return 0;
        }
        int left = 0;
        int right = nums.length - 1;
        int k = 0;
        while (left < right) {
            k = (left + right + 1) / 2;
            if (nums[k] < target)
                left = k;
            else if (nums[k] > target)
                right = k - 1;
            else
                return k;
        }
        if (target <= nums[k]) {
            if (k == 0)
                return 0;
            else if (target > nums[k - 1])
                return k;
            else
                return k - 1;
        } else
            return k + 1;
    }


    /**
     * created by Sven
     * on 2019-12-03
     * <p>
     * 两数之和
     * <p>
     * 给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标。
     * <p>
     * 你可以假设每种输入只会对应一个答案。但是，你不能重复利用这个数组中同样的元素。
     */
    public int[] twoSum(int[] nums, int target) {
        int[] result = new int[2];
        if (nums == null || nums.length < 2)
            return result;
        for (int i = 0; i < nums.length - 1; i++) {
            for (int j = i + 1; j < nums.length; j++) {
                if (nums[i] + nums[j] == target) {
                    result[0] = i;
                    result[1] = j;
                    return result;
                }

            }
        }
        return result;

    }

    public int[] twoSum_2(int[] nums, int target) {
        int[] result = new int[2];
        Map<Integer, Integer> numMap = new HashMap<Integer, Integer>();
        for (int i = 0; i < nums.length; i++) {
            numMap.put(nums[i], i);
        }
        for (int i = 0; i < nums.length; i++) {
            int v = target - nums[i];
            if (numMap.containsKey(v) && i != numMap.get(v)) {
                result[0] = i;
                result[1] = numMap.get(v);
                return result;
            }
        }
        return result;
    }

    /**
     * 小A 和 小B 在玩猜数字。小B 每次从 1, 2, 3 中随机选择一个，小A 每次也从 1, 2, 3 中选择一个猜。他们一共进行三次这个游戏，请返回 小A 猜对了几次？
     * <p>
     *  
     * <p>
     * 输入的guess数组为 小A 每次的猜测，answer数组为 小B 每次的选择。guess和answer的长度都等于3。
     *
     * @param guess
     * @param answer
     * @return
     */
    public int game(int[] guess, int[] answer) {
        if (guess == null || answer == null || guess.length == 0 || answer.length == 0)
            return 0;
        int length = Math.min(guess.length, answer.length);
        int count = 0;
        for (int i = 0; i < length; i++) {
            if (guess[i] == answer[i])
                count++;
        }
        return count;
    }

    /**
     * 给你一个整数数组 nums，请你返回其中位数为 偶数 的数字的个数。
     * <p>
     * 输入：nums = [12,345,2,6,7896]
     * 输出：2
     * 解释：
     * 12 是 2 位数字（位数为偶数） 
     * 345 是 3 位数字（位数为奇数）  
     * 2 是 1 位数字（位数为奇数） 
     * 6 是 1 位数字 位数为奇数） 
     * 7896 是 4 位数字（位数为偶数）  
     * 因此只有 12 和 7896 是位数为偶数的数字
     *
     * @param nums
     * @return
     */
    public int findNumbers(int[] nums) {
        if (nums == null || nums.length == 0)
            return 0;
        int count = 0;
        for (int i = 0; i < nums.length; i++) {
            int tmp = nums[i];
            String tmpStr = String.valueOf(tmp);
            if (tmpStr.length() % 2 == 0)
                count++;
        }
        return count;
    }

    /**
     * 只出现一次的数字
     * <p>
     * 给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。
     * <p>
     * 说明：
     * <p>
     * 你的算法应该具有线性时间复杂度。 你可以不使用额外空间来实现吗？
     *
     * @param nums
     * @return
     */
    public int singleNumber(int[] nums) {
        if (nums == null || nums.length == 0)
            return 0;
        int result = 0;
        for (int i : nums) {
            result ^= i;
        }
        return result;
    }

    /**
     * 两数之和 II - 输入有序数组
     * <p>
     * 给定一个已按照升序排列 的有序数组，找到两个数使得它们相加之和等于目标数。
     * <p>
     * 函数应该返回这两个下标值 index1 和 index2，其中 index1 必须小于 index2。
     * <p>
     * 说明:
     * <p>
     * 返回的下标值（index1 和 index2）不是从零开始的。
     * 你可以假设每个输入只对应唯一的答案，而且你不可以重复使用相同的元素。
     *
     * @param numbers
     * @param target
     * @return
     */
    public int[] twoSum_167(int[] numbers, int target) {
        if (numbers == null || numbers.length < 2)
            return null;
        int left = 0, right = numbers.length - 1;
        while (left < right) {
            if (numbers[left] + numbers[right] == target)
                return new int[]{left + 1, right + 1};
            else if (numbers[left] + numbers[right] > target)
                right--;
            else
                left++;
        }
        return null;
    }

    /**
     * 多数元素
     * <p>
     * 给定一个大小为 n 的数组，找到其中的多数元素。多数元素是指在数组中出现次数大于 ⌊ n/2 ⌋ 的元素。
     * <p>
     * 你可以假设数组是非空的，并且给定的数组总是存在多数元素。
     *
     * @param nums
     * @return
     */
    public int majorityElement(int[] nums) {
        if (nums == null || nums.length == 0)
            return 0;
        Arrays.sort(nums);
        return nums[nums.length / 2];
    }

    /**
     * 打家劫舍
     * <p>
     * 你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。
     * <p>
     * 给定一个代表每个房屋存放金额的非负整数数组，计算你在不触动警报装置的情况下，能够偷窃到的最高金额。
     *
     * @param nums
     * @return
     */

    public int rob(int[] nums) {
        int preMax = 0, currentMax = 0;
        if (nums == null)
            return 0;
        for (int i = 0; i < nums.length; i++) {
            int tmp = currentMax;
            currentMax = Math.max(preMax + nums[i], currentMax);
            preMax = tmp;
        }
        return currentMax;
    }

    /**
     * 面试题40. 最小的k个数
     * 输入整数数组 arr ，找出其中最小的 k 个数。例如，输入4、5、1、6、2、7、3、8这8个数字，则最小的4个数字是1、2、3、4。
     *
     * @param arr
     * @param k
     * @return
     */
    public int[] getLeastNumbers(int[] arr, int k) {
        if (arr == null || arr.length == 0)
            return new int[0];
        Arrays.sort(arr);
        if (arr.length <= k)
            return arr;
        return Arrays.copyOfRange(arr, 0, k);
    }

    private int[] quickSearch(int[] nums, int lo, int hi, int k) {
        // 每快排切分1次，找到排序后下标为j的元素，如果j恰好等于k就返回j以及j左边所有的数；
        int j = partition(nums, lo, hi);
        if (j == k)
            return Arrays.copyOf(nums, j + 1);
        // 否则根据下标j与k的大小关系来决定继续切分左段还是右段。
        return j > k ? quickSearch(nums, lo, j - 1, k) : quickSearch(nums, j + 1, hi, k);
    }

    // 快排切分，返回下标j，使得比nums[j]小的数都在j的左边，比nums[j]大的数都在j的右边。
    private int partition(int[] nums, int lo, int hi) {
        int v = nums[lo];
        int i = lo, j = hi + 1;
        while (true) {
            while (++i <= hi && nums[i] < v) ;
            while (--j >= lo && nums[j] > v) ;
            if (i >= j)
                break;
            int t = nums[j];
            nums[j] = nums[i];
            nums[i] = t;
        }
        nums[lo] = nums[j];
        nums[j] = v;
        return j;
    }

    /**
     * 使数组唯一的最小增量
     * <p>
     * 给定整数数组 A，每次 move 操作将会选择任意 A[i]，并将其递增 1。
     * <p>
     * 返回使 A 中的每个值都是唯一的最少操作次数。
     * <p>
     * 提示：
     * <p>
     * 0 <= A.length <= 40000
     * 0 <= A[i] < 40000
     *
     * @param A
     * @return
     */
    public int minIncrementForUnique(int[] A) {
        if (A == null || A.length == 0)
            return 0;
        // 极限情况，A中有40000个40000，需要一个80000大小的数组
        int[] count = new int[80000];
        // repeatCount 统计当前重复的个数
        int result = 0, repeatCount = 0;
        for (int x : A)
            count[x]++;
        for (int i = 0; i < count.length; i++) {
            if (count[i] > 1) {
                repeatCount += (count[i] - 1);
                result = result - (count[i] - 1) * i;
            }
            if (repeatCount > 0 && count[i] == 0) {
                result += i;
                repeatCount--;
            }
            if (repeatCount == 0 && i > 40000)
                break;

        }
        return result;
    }

    /**
     * 914. 卡牌分组
     * 给定一副牌，每张牌上都写着一个整数。
     * <p>
     * 此时，你需要选定一个数字 X，使我们可以将整副牌按下述规则分成 1 组或更多组：
     * <p>
     * 每组都有 X 张牌。
     * 组内所有的牌上都写着相同的整数。
     * 仅当你可选的 X >= 2 时返回 true。
     *
     * @param deck
     * @return
     */
    public boolean hasGroupsSizeX(int[] deck) {
        if (deck == null || deck.length <= 1)
            return false;
        Map<Integer, Integer> digitalStatistic = new HashMap<>();
        for (int n : deck) {
            if (digitalStatistic.containsKey(n)) {
                digitalStatistic.put(n, digitalStatistic.get(n) + 1);
            } else {
                digitalStatistic.put(n, 1);
            }
        }
        int x = 2;
        boolean hasX = true;
        while (true) {
            for (Integer i : digitalStatistic.keySet()) {
                if (digitalStatistic.get(i) < x)
                    return false;
                if (digitalStatistic.get(i) % x != 0) {
                    hasX = false;
                    break;
                }
            }
            if (hasX)
                return true;
            hasX = true;
            x++;
        }
    }

    /**
     * 820. 单词的压缩编码
     * <p>
     * 给定一个单词列表，我们将这个列表编码成一个索引字符串 S 与一个索引列表 A。
     * <p>
     * 例如，如果这个列表是 ["time", "me", "bell"]，我们就可以将其表示为 S = "time#bell#" 和 indexes = [0, 2, 5]。
     * <p>
     * 对于每一个索引，我们可以通过从字符串 S 中索引的位置开始读取字符串，直到 "#" 结束，来恢复我们之前的单词列表。
     * <p>
     * 那么成功对给定单词列表进行编码的最小字符串长度是多少呢？
     *
     * @param words
     * @return
     */
    public int minimumLengthEncoding(String[] words) {
        if (words == null || words.length == 0)
            return 0;
        Set<String> wordsSet = new HashSet<>(Arrays.asList(words));
        for (String word : words) {
            for (int i = 1; i < word.length(); i++) {
                wordsSet.remove(word.substring(i));
            }
        }

        int result = 0;
        for (String word : wordsSet) {
            result += word.length() + 1;
        }
        return result;
    }

    /**
     * 1162. 地图分析
     * <p>
     * 你现在手里有一份大小为 N x N 的『地图』（网格） grid，上面的每个『区域』（单元格）都用 0 和 1 标记好了。其中 0 代表海洋，1 代表陆地，你知道距离陆地区域最远的海洋区域是是哪一个吗？请返回该海洋区域到离它最近的陆地区域的距离。
     * <p>
     * 我们这里说的距离是『曼哈顿距离』（ Manhattan Distance）：(x0, y0) 和 (x1, y1) 这两个区域之间的距离是 |x0 - x1| + |y0 - y1| 。
     * <p>
     * 如果我们的地图上只有陆地或者海洋，请返回 -1。
     *
     * @param grid
     * @return
     */
    public int maxDistance(int[][] grid) {
        if (grid == null || grid.length == 0)
            return 0;
        int result = -1;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[i].length; j++) {
                if (grid[i][j] == 0) {
                    result = Math.max(result, getMinLandDistance(i, j, grid));

                }
            }
        }
        return result;
    }

    private int getMinLandDistance(int x, int y, int[][] grid) {
        if (grid == null || grid.length == 0)
            return 0;
        int distance = 1;
        while (true) {
            for (int n = 0; n <= distance; n++) {
                if ((x - n >= 0) && (y - distance + n >= 0) && grid[x - n][y - distance + n] == 1)
                    return distance;
                if ((x - n >= 0) && (y + distance - n < grid[x - n].length) && grid[x - n][y + distance - n] == 1)
                    return distance;
                if ((x + n < grid.length) && (y - distance + n >= 0) && grid[x + n][y - distance + n] == 1)
                    return distance;
                if ((x + n < grid.length) && (y + distance - n < grid[x + n].length) && grid[x + n][y + distance - n] == 1)
                    return distance;
            }
            if (distance > (x + y) && distance > (grid.length - x + y) && distance > (x + grid[x].length - y) && distance > (grid.length + grid[x].length - x - y))
                return -1;
            distance++;
        }
    }

    /**
     * 912. 排序数组
     * <p>
     * 给你一个整数数组 nums，将该数组升序排列。
     *
     * @param nums
     * @return
     */
    public int[] sortArray(int[] nums) {
        Arrays.sort(nums);
        return nums;
    }

    /**
     * 289. 生命游戏
     * <p>
     * 根据 百度百科 ，生命游戏，简称为生命，是英国数学家约翰·何顿·康威在 1970 年发明的细胞自动机。
     * <p>
     * 给定一个包含 m × n 个格子的面板，每一个格子都可以看成是一个细胞。每个细胞都具有一个初始状态：1 即为活细胞（live），或 0 即为死细胞（dead）。每个细胞与其八个相邻位置（水平，垂直，对角线）的细胞都遵循以下四条生存定律：
     * <p>
     * 如果活细胞周围八个位置的活细胞数少于两个，则该位置活细胞死亡；
     * 如果活细胞周围八个位置有两个或三个活细胞，则该位置活细胞仍然存活；
     * 如果活细胞周围八个位置有超过三个活细胞，则该位置活细胞死亡；
     * 如果死细胞周围正好有三个活细胞，则该位置死细胞复活；
     * 根据当前状态，写一个函数来计算面板上所有细胞的下一个（一次更新后的）状态。下一个状态是通过将上述规则同时应用于当前状态下的每个细胞所形成的，其中细胞的出生和死亡是同时发生的。
     *
     * @param board
     */
    public void gameOfLife(int[][] board) {
        if (board == null || board.length == 0)
            return;
        int[] neighbors = {0, 1, -1};
        int rows = board.length;
        int cols = board[0].length;

        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                int liveNeighbors = 0;
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        if (!(neighbors[i] == 0 && neighbors[j] == 0)) {
                            int r = (row + neighbors[i]);
                            int c = (col + neighbors[j]);
                            if ((r < rows && r >= 0) && (c < cols && c >= 0) && (Math.abs(board[r][c]) == 1)) {
                                liveNeighbors += 1;
                            }
                        }
                    }
                }
                if ((board[row][col] == 1) && (liveNeighbors < 2 || liveNeighbors > 3)) {
                    board[row][col] = -1;
                }
                if (board[row][col] == 0 && liveNeighbors == 3) {
                    board[row][col] = 2;
                }
            }
        }

        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                if (board[row][col] > 0)
                    board[row][col] = 1;
                else
                    board[row][col] = 0;
            }
        }
    }

    /**
     * 42. 接雨水
     * 给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。
     *
     * @param height
     * @return
     */
    public int trap(int[] height) {
        if (height == null || height.length == 0)
            return 0;
        int heightSize = height.length;
        int result = 0;
        int[] leftMax = new int[heightSize], rightMax = new int[heightSize];
        leftMax[0] = height[0];
        for (int i = 1; i < heightSize; i++) {
            leftMax[i] = Math.max(height[i], leftMax[i - 1]);
        }
        rightMax[heightSize - 1] = height[heightSize - 1];
        for (int i = heightSize - 2; i >= 0; i--) {
            rightMax[i] = Math.max(height[i], rightMax[i + 1]);
        }
        for (int i = 1; i < heightSize - 1; i++) {
            result += Math.min(leftMax[i], rightMax[i]) - height[i];
        }
        return result;

    }

    /**
     * 72. 编辑距离
     * 给你两个单词 word1 和 word2，请你计算出将 word1 转换成 word2 所使用的最少操作数 。
     * <p>
     * 你可以对一个单词进行如下三种操作：
     * <p>
     * 插入一个字符
     * 删除一个字符
     * 替换一个字符
     *
     * @param word1
     * @param word2
     * @return
     */
    public int minDistance(String word1, String word2) {
        if (word1 == null && word2 == null)
            return 0;
        if (word1 == null)
            return word2.length();
        if (word2 == null)
            return word1.length();

        int lengthWord1 = word1.length();
        int lengthWord2 = word2.length();

        if (lengthWord1 * lengthWord2 == 0)
            return lengthWord1 + lengthWord2;

        int[][] dp = new int[lengthWord1 + 1][lengthWord2 + 1];
        for (int i = 0; i < lengthWord1 + 1; i++) {
            dp[i][0] = i;
        }
        for (int j = 0; j < lengthWord2 + 1; j++) {
            dp[0][j] = j;
        }
        for (int i = 1; i < lengthWord1 + 1; i++) {
            for (int j = 1; j < lengthWord2 + 1; j++) {
                int left = dp[i - 1][j] + 1;
                int down = dp[i][j - 1] + 1;
                int left_down = dp[i - 1][j - 1];
                if (word1.charAt(i - 1) != word2.charAt(j - 1))
                    left_down += 1;
                dp[i][j] = Math.min(left, Math.min(down, left_down));
            }
        }
        return dp[lengthWord1][lengthWord2];
    }

    /**
     * 面试题 01.07. 旋转矩阵
     * 给你一幅由 N × N 矩阵表示的图像，其中每个像素的大小为 4 字节。请你设计一种算法，将图像旋转 90 度。
     * <p>
     * 不占用额外内存空间能否做到？
     *
     * @param matrix
     */
    public void rotate(int[][] matrix) {
        if (matrix == null || matrix.length == 0)
            return;
        int n = matrix.length;
        for (int i = 0; i < n - 1; i++) {
            for (int j = i + 1; j < n; j++) {
                int tmp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = tmp;
            }
        }
        int mid = n >> 1;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < mid; j++) {
                int tmp = matrix[i][j];
                matrix[i][j] = matrix[i][n - 1 - j];
                matrix[i][n - 1 - j] = tmp;
            }
        }
    }

    /**
     * 542. 01 矩阵
     * 给定一个由 0 和 1 组成的矩阵，找出每个元素到最近的 0 的距离。
     * <p>
     * 两个相邻元素间的距离为 1 。
     * 注意:
     * <p>
     * 给定矩阵的元素个数不超过 10000。
     * 给定矩阵中至少有一个元素是 0。
     * 矩阵中的元素只在四个方向上相邻: 上、下、左、右。
     *
     * @param matrix
     * @return
     */
    public int[][] updateMatrix(int[][] matrix) {
        Queue<int[]> queue = new LinkedList<>();
        int m = matrix.length, n = matrix[0].length;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == 0) {
                    queue.offer(new int[]{i, j});
                } else {
                    matrix[i][j] = -1;
                }
            }
        }
        int[] dx = new int[]{-1, 1, 0, 0};
        int[] dy = new int[]{0, 0, -1, 1};
        while (!queue.isEmpty()) {
            int[] point = queue.poll();
            int x = point[0], y = point[1];
            for (int i = 0; i < 4; i++) {
                int newX = x + dx[i];
                int newY = y + dy[i];
                // 如果四邻域的点是 -1，表示这个点是未被访问过的 1
                // 所以这个点到 0 的距离就可以更新成 matrix[x][y] + 1。
                if (newX >= 0 && newX < m && newY >= 0 && newY < n
                        && matrix[newX][newY] == -1) {
                    matrix[newX][newY] = matrix[x][y] + 1;
                    queue.offer(new int[]{newX, newY});
                }
            }
        }
        return matrix;
    }

    /**
     * 56. 合并区间
     * 给出一个区间的集合，请合并所有重叠的区间。
     *
     * @param intervals
     * @return
     */
    public int[][] merge(int[][] intervals) {
        if (intervals == null || intervals.length == 0)
            return new int[0][0];
        Arrays.parallelSort(intervals, Comparator.comparingInt(x -> x[0]));
        LinkedList<int[]> list = new LinkedList<>();
        for (int i = 0; i < intervals.length; i++) {
            if (list.size() == 0 || list.getLast()[1] < intervals[i][0]) {
                list.add(intervals[i]);
            } else {
                list.getLast()[1] = Math.max(list.getLast()[1], intervals[i][1]);
            }
        }
        int[][] res = new int[list.size()][2];
        int index = 0;
        while (!list.isEmpty()) {
            res[index++] = list.removeFirst();
        }
        return res;
    }

    /**
     * 55. 跳跃游戏
     * 给定一个非负整数数组，你最初位于数组的第一个位置。
     * <p>
     * 数组中的每个元素代表你在该位置可以跳跃的最大长度。
     * <p>
     * 判断你是否能够到达最后一个位置。
     *
     * @param nums
     * @return
     */
    public boolean canJump(int[] nums) {
        int rightMost = 0;
        for (int i = 0; i < nums.length; i++) {
            if (i <= rightMost) {
                rightMost = Math.max(rightMost, i + nums[i]);
                if (rightMost >= nums.length - 1)
                    return true;
            } else
                return false;
        }
        return false;
    }

    /**
     * 11. 盛最多水的容器
     * 给你 n 个非负整数 a1，a2，...，an，每个数代表坐标中的一个点 (i, ai) 。在坐标内画 n 条垂直线，垂直线 i 的两个端点分别为 (i, ai) 和 (i, 0)。找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。
     * <p>
     * 说明：你不能倾斜容器，且 n 的值至少为 2。
     *
     * @param height
     * @return
     */
    public int maxArea(int[] height) {
        int l = 0, r = height.length - 1;
        int result = 0;
        while (l < r) {
            int area = Math.min(height[l], height[r]) * (r - l);
            result = Math.max(result, area);
            if (height[l] <= height[r]) {
                ++l;
            } else {
                --r;
            }
        }
        return result;
    }

    /**
     * 200. 岛屿数量
     * <p>
     * 给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。
     * <p>
     * 岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。
     * <p>
     * 此外，你可以假设该网格的四条边均被水包围。
     *
     * @param grid
     * @return
     */
    public int numIslands(char[][] grid) {
        if (grid == null || grid.length == 0)
            return 0;
        int nr = grid.length;
        int nc = grid[0].length;
        int num_islands = 0;
        for (int r = 0; r < nr; r++) {
            for (int c = 0; c < nc; c++) {
                if (grid[r][c] == '1') {
                    ++num_islands;
                    numIslandsDFS(grid, r, c);
                }
            }
        }
        return num_islands;
    }

    private void numIslandsDFS(char[][] grid, int r, int c) {
        int nr = grid.length;
        int nc = grid[0].length;

        if (r < 0 || c < 0 || r >= nr || c >= nc || grid[r][c] == '0') {
            return;
        }

        grid[r][c] = '0';
        numIslandsDFS(grid, r - 1, c);
        numIslandsDFS(grid, r + 1, c);
        numIslandsDFS(grid, r, c - 1);
        numIslandsDFS(grid, r, c + 1);
    }

    /**
     * 1248. 统计「优美子数组」
     * 给你一个整数数组 nums 和一个整数 k。
     * 如果某个 连续 子数组中恰好有 k 个奇数数字，我们就认为这个子数组是「优美子数组」。
     * 请返回这个数组中「优美子数组」的数目。
     *
     * @param nums
     * @param k
     * @return
     */
    public int numberOfSubarrays(int[] nums, int k) {
        if (nums == null || nums.length == 0 || nums.length < k) return 0;
        int left = 0, right = 0;
        int count = 0;
        int res = 0;
        int preEven = 0;
        while (right < nums.length) {
            if (count < k) {
                if (nums[right] % 2 != 0) count++;
                right++;
            }
            if (count == k) {
                preEven = 0;
                while (count == k) {
                    res++;
                    if (nums[left] % 2 != 0) count--;
                    left++;
                    preEven++;
                }
            } else res += preEven;
        }
        return res;
    }

    /**
     * 46. 全排列
     * <p>
     * 给定一个 没有重复 数字的序列，返回其所有可能的全排列。
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> permute(int[] nums) {
        if (nums == null || nums.length == 0)
            return null;
        List<List<Integer>> res = new LinkedList<>();
        ArrayList<Integer> output = new ArrayList<Integer>();
        for (int num : nums)
            output.add(num);
        int n = nums.length;
        backtrack(n, output, res, 0);
        return res;
    }

    private void backtrack(int n, ArrayList<Integer> output, List<List<Integer>> res, int first) {
        if (first == n)
            res.add(new ArrayList<Integer>(output));
        for (int i = first; i < n; i++) {
            Collections.swap(output, first, i);
            backtrack(n, output, res, first + 1);
            Collections.swap(output, first, i);
        }
    }

    /**
     * 33. 搜索旋转排序数组
     * <p>
     * <p>
     * 假设按照升序排序的数组在预先未知的某个点上进行了旋转。
     * <p>
     * ( 例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] )。
     * <p>
     * 搜索一个给定的目标值，如果数组中存在这个目标值，则返回它的索引，否则返回 -1 。
     * <p>
     * 你可以假设数组中不存在重复的元素。
     * <p>
     * 你的算法时间复杂度必须是 O(log n) 级别。
     *
     * @param nums
     * @param target
     * @return
     */
    public int search(int[] nums, int target) {
        if (nums == null || nums.length == 0)
            return -1;
        int length = nums.length;
        if (length == 1)
            return nums[0] == target ? 0 : -1;
        int left = 0, right = length - 1;
        while (left <= right) {
            int mid = (left + right) / 2;
            if (nums[mid] == target) return mid;
            if (nums[0] <= nums[mid]) {
                if (nums[0] <= target && target < nums[mid]) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            } else {
                if (nums[mid] < target && target <= nums[length - 1])
                    left = mid + 1;
                else
                    right = mid - 1;
            }
        }
        return -1;
    }


    /**
     * 面试题56 - I. 数组中数字出现的次数
     * <p>
     * 一个整型数组 nums 里除两个数字之外，其他数字都出现了两次。请写程序找出这两个只出现一次的数字。要求时间复杂度是O(n)，空间复杂度是O(1)。
     *
     * @param nums
     * @return
     */
    public int[] singleNumbers(int[] nums) {
        int sum = 0;
        for (int i = 0; i < nums.length; i++) {
            sum ^= nums[i];
        }
        int first = 1;
        while ((sum & first) == 0)
            first = first << 1;
        int[] result = new int[2];
        for (int i = 0; i < nums.length; i++) {
            //将数组分类。
            if ((nums[i] & first) == 0) {
                result[0] ^= nums[i];
            } else {
                result[1] ^= nums[i];
            }
        }
        return result;
    }

    /**
     * 1095. 山脉数组中查找目标值
     * （这是一个 交互式问题 ）
     * <p>
     * 给你一个 山脉数组 mountainArr，请你返回能够使得 mountainArr.get(index) 等于 target 最小 的下标 index 值。
     * <p>
     * 如果不存在这样的下标 index，就请返回 -1。
     * <p>
     *  
     * <p>
     * 何为山脉数组？如果数组 A 是一个山脉数组的话，那它满足如下条件：
     * <p>
     * 首先，A.length >= 3
     * <p>
     * 其次，在 0 < i < A.length - 1 条件下，存在 i 使得：
     * <p>
     * A[0] < A[1] < ... A[i-1] < A[i]
     * A[i] > A[i+1] > ... > A[A.length - 1]
     *  
     * <p>
     * 你将 不能直接访问该山脉数组，必须通过 MountainArray 接口来获取数据：
     * <p>
     * MountainArray.get(k) - 会返回数组中索引为k 的元素（下标从 0 开始）
     * MountainArray.length() - 会返回该数组的长度
     *  
     * <p>
     * 注意：
     * <p>
     * 对 MountainArray.get 发起超过 100 次调用的提交将被视为错误答案。此外，任何试图规避判题系统的解决方案都将会导致比赛资格被取消。
     * <p>
     * 为了帮助大家更好地理解交互式问题，我们准备了一个样例 “答案”：https://leetcode-cn.com/playground/RKhe3ave，请注意这 不是一个正确答案。
     *
     * @param target
     * @param mountainArr
     * @return
     */
    public int findInMountainArray(int target, MountainArray mountainArr) {
        if (mountainArr == null || mountainArr.length() == 0)
            return -1;
        int left = 0, right = mountainArr.length() - 1;
        int mid = left + ((right - left) >> 1);
        while (left < right) {
            if (mountainArr.get(mid) < mountainArr.get(mid + 1))
                left = mid + 1;
            else
                right = mid;
            mid = left + ((right - left) >> 1);
        }
        left = 0;
        int tmp = right;
        int midLeft = (left + right) / 2;
        while (left < right) {
            if (mountainArr.get(midLeft) == target) return midLeft;
            if (mountainArr.get(midLeft) > target)
                right = midLeft;
            else
                left = midLeft + 1;
            midLeft = (left + right) / 2;
        }
        left = tmp;
        right = mountainArr.length();
        int midRight = (left + right) / 2;
        while (left < right) {
            if (mountainArr.get(midRight) == target) return midRight;
            if (mountainArr.get(midRight) > target)
                left = midRight + 1;
            else
                right = midRight;
            midRight = (left + right) / 2;
        }
        return -1;
    }

    /**
     * 1313. 解压缩编码列表
     * <p>
     * 给你一个以行程长度编码压缩的整数列表 nums 。
     * <p>
     * 考虑每对相邻的两个元素 [freq, val] = [nums[2*i], nums[2*i+1]] （其中 i >= 0 ），每一对都表示解压后子列表中有 freq 个值为 val 的元素，你需要从左到右连接所有子列表以生成解压后的列表。
     * <p>
     * 请你返回解压后的列表。
     *
     * @param nums
     * @return
     */
    public int[] decompressRLElist(int[] nums) {
        if (nums == null || nums.length < 2)
            return new int[0];
        ArrayList<Integer> list = new ArrayList<>();
        int i = 0;
        while (2 * i < nums.length) {
            for (int n = 0; n < nums[2 * i]; n++) {
                list.add(nums[2 * i + 1]);
            }
            i++;
        }
        int[] resultArray = new int[list.size()];
        for (int n = 0; n < list.size(); n++) {
            resultArray[n] = list.get(n);
        }
        return resultArray;
    }

    /**
     * 1365. 有多少小于当前数字的数字
     * <p>
     * 给你一个数组 nums，对于其中每个元素 nums[i]，请你统计数组中比它小的所有数字的数目。
     * <p>
     * 换而言之，对于每个 nums[i] 你必须计算出有效的 j 的数量，其中 j 满足 j != i 且 nums[j] < nums[i] 。
     * <p>
     * 以数组形式返回答案。
     *
     * @param nums
     * @return
     */
    public int[] smallerNumbersThanCurrent(int[] nums) {
        if (nums == null || nums.length == 0)
            return new int[0];
        int[] temp = new int[101];
        for (int i = 0; i < nums.length; i++) {
            temp[nums[i]] += 1;
        }
        for (int i = 1; i < temp.length; i++) {
            temp[i] = temp[i - 1] + temp[i];
        }
        int[] result = new int[nums.length];
        for (int i = 0; i < result.length; i++) {
            if (nums[i] == 0)
                result[i] = 0;
            else {
                result[i] = temp[nums[i] - 1];
            }
        }
        return result;
    }

    /**
     * 3. 无重复字符的最长子串
     * 给定一个字符串，请你找出其中不含有重复字符的 最长子串 的长度。
     *
     * @param s
     * @return
     */
    public int lengthOfLongestSubstring(String s) {
        Set<Character> occ = new HashSet<>();
        int n = s.length();
        int rk = -1, ans = 0;
        for (int i = 0; i < n; ++i) {
            if (i != 0)
                occ.remove(s.charAt(i - 1));
            while (rk + 1 < n && !occ.contains(s.charAt(rk + 1))) {
                occ.add(s.charAt(rk + 1));
                ++rk;
            }
            ans = Math.max(ans, rk - i + 1);
        }
        return ans;

    }

    /**
     * 1389. 按既定顺序创建目标数组
     * 给你两个整数数组 nums 和 index。你需要按照以下规则创建目标数组：
     * <p>
     * 目标数组 target 最初为空。
     * 按从左到右的顺序依次读取 nums[i] 和 index[i]，在 target 数组中的下标 index[i] 处插入值 nums[i] 。
     * 重复上一步，直到在 nums 和 index 中都没有要读取的元素。
     * 请你返回目标数组。
     * <p>
     * 题目保证数字插入位置总是存在。
     *
     * @param nums
     * @param index
     * @return
     */
    public int[] createTargetArray(int[] nums, int[] index) {
        if (nums == null || nums.length == 0 || index == null || index.length == 0)
            return new int[0];
        ArrayList<Integer> resultList = new ArrayList<>();
        for (int i = 0; i < index.length && i < nums.length; i++) {
            resultList.add(index[i], nums[i]);
        }
        int[] result = new int[resultList.size()];
        for (int i = 0; i < resultList.size(); i++) {
            result[i] = resultList.get(i);
        }
        return result;
    }

    /**
     * 45. 跳跃游戏 II
     * 给定一个非负整数数组，你最初位于数组的第一个位置。
     * <p>
     * 数组中的每个元素代表你在该位置可以跳跃的最大长度。
     * <p>
     * 你的目标是使用最少的跳跃次数到达数组的最后一个位置。
     *
     * @param nums
     * @return
     */
    public int jump(int[] nums) {
        if (nums == null || nums.length == 0)
            return 0;
        int position = nums.length - 1;
        int step = 0;
        while (position > 0) {
            for (int i = 0; i < position; i++) {
                if (i + nums[i] >= position) {
                    position = i;
                    step++;
                    break;
                }
            }
        }
        return step;
    }

    public int jump2(int[] nums) {
        if (nums == null || nums.length == 0)
            return 0;
        int maxPosition = 0;
        int step = 0;
        int end = 0;
        for (int i = 0; i < nums.length; i++) {
            maxPosition = Math.max(maxPosition, i + nums[i]);
            if (end == i) {
                end = maxPosition;
                step++;
            }
        }
        return step;
    }

    /**
     * 221. 最大正方形
     * <p>
     * 在一个由 0 和 1 组成的二维矩阵内，找到只包含 1 的最大正方形，并返回其面积。
     *
     * @param matrix
     * @return
     */
    public int maximalSquare(char[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0)
            return 0;
        int maxSide = 0;
        int rows = matrix.length, columns = matrix[0].length;
        int[][] dp = new int[rows][columns];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                if (matrix[i][j] == '1') {
                    if (i == 0 || j == 0) {
                        dp[i][j] = 1;
                    } else {
                        dp[i][j] = Math.min(Math.min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]) + 1;
                    }
                    maxSide = Math.max(maxSide, dp[i][j]);
                }
            }
        }
        return maxSide * maxSide;
    }

    /**
     * 1431. 拥有最多糖果的孩子
     * <p>
     * 给你一个数组 candies 和一个整数 extraCandies ，其中 candies[i] 代表第 i 个孩子拥有的糖果数目。
     * <p>
     * 对每一个孩子，检查是否存在一种方案，将额外的 extraCandies 个糖果分配给孩子们之后，此孩子有 最多 的糖果。注意，允许有多个孩子同时拥有 最多 的糖果数目。
     *
     * @param candies
     * @param extraCandies
     * @return
     */
    public List<Boolean> kidsWithCandies(int[] candies, int extraCandies) {
        if (candies == null || candies.length == 0)
            return new ArrayList<>();
        List<Boolean> resultList = new ArrayList<>();
        int max = 0;
        for (int candy : candies) {
            if (max < candy)
                max = candy;
        }
        for (int i = 0; i < candies.length; i++) {
            if (candies[i] + extraCandies >= max)
                resultList.add(true);
            else
                resultList.add(false);
        }
        return resultList;
    }

    /**
     * LCP 06. 拿硬币
     * <p>
     * 桌上有 n 堆力扣币，每堆的数量保存在数组 coins 中。我们每次可以选择任意一堆，拿走其中的一枚或者两枚，求拿完所有力扣币的最少次数。
     *
     * @param coins
     * @return
     */
    public int minCount(int[] coins) {
        if (coins == null || coins.length == 0)
            return 0;
        int result = 0;
        for (int i = 0; i < coins.length; i++) {
            result += coins[i] / 2;
            if (coins[i] % 2 != 0)
                result++;

        }
        return result;
    }

    /**
     * @param nums
     * @param k
     * @return
     */
    public int subarraySum(int[] nums, int k) {
        if (nums == null && nums.length == 0)
            return 0;
        int count = 0;
        for (int start = 0; start < nums.length; start++) {
            int sum = 0;
            for (int end = start; end < nums.length; end++) {
                sum += nums[end];
                if (sum == k)
                    count++;
            }
        }
        return count;
    }

    public int subarraySumV2(int[] nums, int k) {
        int count = 0, pre = 0;
        HashMap<Integer, Integer> mp = new HashMap<>();
        mp.put(0, 1);
        for (int i = 0; i < nums.length; i++) {
            pre += nums[i];
            if (mp.containsKey(pre - k))
                count += mp.get(pre - k);
            mp.put(pre, mp.getOrDefault(pre, 0) + 1);
        }
        return count;
    }

    /**
     * 210. 课程表 II
     * <p>
     * 现在你总共有 n 门课需要选，记为 0 到 n-1。
     * <p>
     * 在选修某些课程之前需要一些先修课程。 例如，想要学习课程 0 ，你需要先完成课程 1 ，我们用一个匹配来表示他们: [0,1]
     * <p>
     * 给定课程总量以及它们的先决条件，返回你为了学完所有课程所安排的学习顺序。
     * <p>
     * 可能会有多个正确的顺序，你只要返回一种就可以了。如果不可能完成所有课程，返回一个空数组。
     *
     * @param numCourses
     * @param prerequisites
     * @return
     */
    public int[] findOrder(int numCourses, int[][] prerequisites) {
        if (numCourses == 0) return new int[0];
        int[] inDegrees = new int[numCourses];
        for (int[] p : prerequisites) {
            inDegrees[p[0]]++;
        }
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < inDegrees.length; i++) {
            if (inDegrees[i] == 0) queue.offer(i);
        }
        int count = 0;
        int[] res = new int[numCourses];
        while (!queue.isEmpty()) {
            int curr = queue.poll();
            res[count++] = curr;
            for (int[] p : prerequisites) {
                if (p[1] == curr) {
                    inDegrees[p[0]]--;
                    if (inDegrees[p[0]] == 0) queue.offer(p[0]);
                }
            }
        }
        if (count == numCourses) return res;
        return new int[0];
    }

    /**
     * 152. 乘积最大子数组
     * <p>
     * 给你一个整数数组 nums ，请你找出数组中乘积最大的连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。
     *
     * @param nums
     * @return
     */
    public int maxProduct(int[] nums) {
        if (nums == null || nums.length == 0)
            return 0;
        int maxF = nums[0], minF = nums[0], ans = nums[0];
        for (int i = 1; i < nums.length; i++) {
            int mx = maxF, mn = minF;
            maxF = Math.max(mx * nums[i], Math.max(nums[i], mn * nums[i]));
            minF = Math.min(mn * nums[i], Math.min(nums[i], mx * nums[i]));
            ans = Math.max(maxF, ans);
        }
        return ans;
    }

    /**
     * 4. 寻找两个正序数组的中位数
     * <p>
     * 给定两个大小为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。
     * <p>
     * 请你找出这两个正序数组的中位数，并且要求算法的时间复杂度为 O(log(m + n))。
     * <p>
     * 你可以假设 nums1 和 nums2 不会同时为空。
     *
     * @param nums1
     * @param nums2
     * @return
     */
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        if (nums1 == null && nums2 == null)
            return 0.0;
        if (nums1 == null || nums1.length == 0) {
            if (nums2.length == 0)
                return 0.0;
            else
                return nums2[nums2.length / 2];
        }
        if (nums2 == null || nums2.length == 0) {
            if (nums1.length == 0)
                return 0.0;
            else
                return nums1[nums1.length / 2];
        }
        if (nums1.length > nums2.length)
            return findMedianSortedArrays(nums2, nums1);
        int m = nums1.length;
        int n = nums2.length;
        int left = 0, right = m, ansi = -1;
        int median1 = 0, median2 = 0;

        while (left <= right) {
            int i = (left + right) / 2;
            int j = (m + n + 1) / 2 - i;

            int nums_im1 = (i == 0 ? Integer.MIN_VALUE : nums1[i - 1]);
            int nums_i = (i == m ? Integer.MAX_VALUE : nums1[i]);
            int nums_jm1 = (j == 0 ? Integer.MIN_VALUE : nums2[j - 1]);
            int nums_j = (j == n ? Integer.MAX_VALUE : nums2[j]);

            if (nums_im1 <= nums_j) {
                ansi = i;
                median1 = Math.max(nums_im1, nums_jm1);
                median2 = Math.min(nums_i, nums_j);
                left = i + 1;
            } else {
                right = i - 1;
            }
        }
        return (m + n) % 2 == 0 ? (median1 + median2) / 2.0 : median1;

    }

    /**
     * 287. 寻找重复数
     * <p>
     * 给定一个包含 n + 1 个整数的数组 nums，其数字都在 1 到 n 之间（包括 1 和 n），可知至少存在一个重复的整数。假设只有一个重复的整数，找出这个重复的数。
     *
     * @param nums
     * @return
     */
    public int findDuplicate(int[] nums) {
        int slow = 0, fast = 0;
        do {
            slow = nums[slow];
            fast = nums[nums[fast]];
        } while (slow != fast);
        slow = 0;
        while (slow != fast) {
            slow = nums[slow];
            fast = nums[fast];
        }
        return slow;
    }

    /**
     * 974. 和可被 K 整除的子数组
     * <p>
     * 给定一个整数数组 A，返回其中元素之和可被 K 整除的（连续、非空）子数组的数目。
     *
     * @param A
     * @param K
     * @return
     */
    public int subarraysDivByK(int[] A, int K) {
        Map<Integer, Integer> record = new HashMap<>();
        record.put(0, 1);
        int sum = 0, ans = 0;
        for (int elem : A) {
            sum += elem;
            int modules = (sum % K + K) % K;
            int same = record.getOrDefault(modules, 0);
            ans += same;
            record.put(modules, same + 1);
        }
        return ans;
    }

    /**
     * 84. 柱状图中最大的矩形
     * 给定 n 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。
     * <p>
     * 求在该柱状图中，能够勾勒出来的矩形的最大面积。
     *
     * @param heights
     * @return
     */

    public int largestRectangleArea(int[] heights) {
        int n = heights.length;
        int[] left = new int[n];
        int[] right = new int[n];

        Stack<Integer> mono_stack = new Stack<Integer>();
        for (int i = 0; i < n; ++i) {
            while (!mono_stack.isEmpty() && heights[mono_stack.peek()] >= heights[i]) {
                mono_stack.pop();
            }
            left[i] = (mono_stack.isEmpty() ? -1 : mono_stack.peek());
            mono_stack.push(i);
        }

        mono_stack.clear();
        for (int i = n - 1; i >= 0; --i) {
            while (!mono_stack.isEmpty() && heights[mono_stack.peek()] >= heights[i]) {
                mono_stack.pop();
            }
            right[i] = (mono_stack.isEmpty() ? n : mono_stack.peek());
            mono_stack.push(i);
        }

        int ans = 0;
        for (int i = 0; i < n; ++i) {
            ans = Math.max(ans, (right[i] - left[i] - 1) * heights[i]);
        }
        return ans;
    }

    /**
     * 1460. 通过翻转子数组使两个数组相等
     * 给你两个长度相同的整数数组 target 和 arr 。
     * <p>
     * 每一步中，你可以选择 arr 的任意 非空子数组 并将它翻转。你可以执行此过程任意次。
     * <p>
     * 如果你能让 arr 变得与 target 相同，返回 True；否则，返回 False 。
     *
     * @param target
     * @param arr
     * @return
     */
    public boolean canBeEqual(int[] target, int[] arr) {
        if (target == null && arr == null)
            return true;
        if (target == null || arr == null)
            return false;
        if (target.length != arr.length)
            return false;
        HashMap<Integer, Integer> hashMap = new HashMap();
        for (int i = 0; i < target.length; i++) {
            hashMap.put(target[i], hashMap.getOrDefault(target[i], 0) + 1);
        }
        for (int i = 0; i < arr.length; i++) {
            if (hashMap.containsKey(arr[i])) {
                if (hashMap.get(arr[i]) > 0) {
                    hashMap.put(arr[i], hashMap.get(arr[i]) - 1);
                } else {
                    return false;
                }
            } else
                return false;
        }
        return true;
    }

    /**
     * 1450. 在既定时间做作业的学生人数
     * <p>
     * 给你两个整数数组 startTime（开始时间）和 endTime（结束时间），并指定一个整数 queryTime 作为查询时间。
     * <p>
     * 已知，第 i 名学生在 startTime[i] 时开始写作业并于 endTime[i] 时完成作业。
     * <p>
     * 请返回在查询时间 queryTime 时正在做作业的学生人数。形式上，返回能够使 queryTime 处于区间 [startTime[i], endTime[i]]（含）的学生人数。
     *
     * @param startTime
     * @param endTime
     * @param queryTime
     * @return
     */
    public int busyStudent(int[] startTime, int[] endTime, int queryTime) {
        if (startTime == null || startTime.length == 0 || endTime == null || endTime.length == 0)
            return 0;
        int length = Math.min(startTime.length, endTime.length);
        int count = 0;
        for (int i = 0; i < length; i++) {
            if (queryTime >= startTime[i] && queryTime <= endTime[i])
                count++;
        }
        return count;
    }

    /**
     * 238. 除自身以外数组的乘积
     * 给你一个长度为 n 的整数数组 nums，其中 n > 1，返回输出数组 output ，其中 output[i] 等于 nums 中除 nums[i] 之外其余各元素的乘积。
     *
     * @param nums
     * @return
     */
    public int[] productExceptSelf(int[] nums) {
        if (nums == null || nums.length == 0)
            return new int[0];
        int mulAll = 1;
        int haveZero = 0;
        int[] result = new int[nums.length];
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == 0)
                haveZero++;
            else
                mulAll = mulAll * nums[i];
        }
        if (haveZero >= 2) {
            return result;
        } else if (haveZero == 1) {
            for (int i = 0; i < nums.length; i++) {
                if (nums[i] == 0) {
                    result[i] = mulAll;
                    return result;
                }

            }
        }
        for (int i = 0; i < nums.length; i++) {
            result[i] = mulAll / nums[i];
        }
        return result;
    }


    /**
     * 面试题29. 顺时针打印矩阵
     * <p>
     * 输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字。
     *
     * @param matrix
     * @return
     */
    public int[] spiralOrder(int[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0)
            return new int[0];
        int rows = matrix.length, columns = matrix[0].length;
        int[] order = new int[rows * columns];
        int index = 0;
        int left = 0, right = columns - 1, top = 0, bottom = rows - 1;
        while (left <= right && top <= bottom) {
            for (int column = left; column <= right; column++)
                order[index++] = matrix[top][column];
            for (int row = top + 1; row <= bottom; row++)
                order[index++] = matrix[row][right];
            if (left < right && top < bottom) {
                for (int column = right - 1; column > left; column--)
                    order[index++] = matrix[bottom][column];
                for (int row = bottom; row > top; row--)
                    order[index++] = matrix[row][left];
            }
            left++;
            right--;
            top++;
            bottom--;
        }
        return order;
    }

    /**
     * 128. 最长连续序列
     * <p>
     * 给定一个未排序的整数数组，找出最长连续序列的长度。
     * <p>
     * 要求算法的时间复杂度为 O(n)。
     *
     * @param nums
     * @return
     */
    public int longestConsecutive(int[] nums) {
        Set<Integer> num_set = new HashSet<>();
        for (int num : nums) {
            num_set.add(num);
        }
        int longestSteak = 0;
        for (int num : num_set) {
            if (!num_set.contains(num - 1)) {
                int currentNum = num;
                int currentStreak = 1;
                while (num_set.contains(currentNum + 1)) {
                    currentNum += 1;
                    currentStreak += 1;
                }
                longestSteak = Math.max(longestSteak, currentStreak);
            }
        }
        return longestSteak;
    }

    /**
     * 739. 每日温度
     * <p>
     * 根据每日 气温 列表，请重新生成一个列表，对应位置的输出是需要再等待多久温度才会升高超过该日的天数。如果之后都不会升高，请在该位置用 0 来代替。
     * <p>
     * 例如，给定一个列表 temperatures = [73, 74, 75, 71, 69, 72, 76, 73]，你的输出应该是 [1, 1, 4, 2, 1, 1, 0, 0]。
     * <p>
     * 提示：气温 列表长度的范围是 [1, 30000]。每个气温的值的均为华氏度，都是在 [30, 100] 范围内的整数。
     *
     * @param T
     * @return
     */
    public int[] dailyTemperatures(int[] T) {
        if (T == null)
            return new int[0];
        int length = T.length;
        int[] ans = new int[length];
        Deque<Integer> stack = new LinkedList<Integer>();
        for (int i = 0; i < length; i++) {
            int temperature = T[i];
            while (!stack.isEmpty() && temperature > T[stack.peek()]) {
                int prevIndex = stack.pop();
                ans[prevIndex] = i - prevIndex;
            }
            stack.push(i);
        }
        return ans;

    }

    /**
     * 15. 三数之和
     * 给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有满足条件且不重复的三元组。
     * <p>
     * 注意：答案中不可以包含重复的三元组。
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> threeSum(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> result = new ArrayList<List<Integer>>();
        for (int i = 0; i < nums.length - 2; i++) {
            if (i == 0 || (i > 0 && nums[i] != nums[i - 1])) {
                int left = i + 1, right = nums.length - 1, sum = 0 - nums[i];
                while (left < right) {
                    if (nums[left] + nums[right] == sum) {
                        result.add(Arrays.asList(nums[i], nums[left], nums[right]));
                        while (left < right && nums[left] == nums[left + 1]) left++;
                        while (left < right && nums[right] == nums[right - 1]) right--;
                        left++;
                        right--;
                    } else if (nums[left] + nums[right] < sum) {
                        while (left < right && nums[left] == nums[left + 1]) left++;
                        left++;
                    } else if (nums[left] + nums[right] > sum) {
                        while (left < right && nums[right] == nums[right - 1]) right--;
                        right--;
                    }
                }
            }
        }
        return result;
    }

    public int findBestValue(int[] arr, int target) {
        Arrays.sort(arr);
        int n = arr.length;
        int[] prefix = new int[n + 1];
        for (int i = 1; i <= n; ++i) {
            prefix[i] = prefix[i - 1] + arr[i - 1];
        }
        int l = 0, r = arr[n - 1], ans = -1;
        while (l <= r) {
            int mid = (l + r) / 2;
            int index = Arrays.binarySearch(arr, mid);
            if (index < 0) {
                index = -index - 1;
            }
            int cur = prefix[index] + (n - index) * mid;
            if (cur <= target) {
                ans = mid;
                l = mid + 1;
            } else {
                r = mid - 1;
            }
        }
        int chooseSmall = check(arr, ans);
        int chooseBig = check(arr, ans + 1);
        return Math.abs(chooseSmall - target) <= Math.abs(chooseBig - target) ? ans : ans + 1;
    }

    public int check(int[] arr, int x) {
        int ret = 0;
        for (int num : arr) {
            ret += Math.min(num, x);
        }
        return ret;
    }

    /**
     * 14. 最长公共前缀
     * <p>
     * 编写一个函数来查找字符串数组中的最长公共前缀。
     * <p>
     * 如果不存在公共前缀，返回空字符串 ""。
     *
     * @param strs
     * @return
     */
    public String longestCommonPrefix(String[] strs) {
        if (strs == null || strs.length == 0 || strs[0] == null || strs[0].length() == 0)
            return "";
        int minLength = strs[0].length();
        for (String str : strs) {
            minLength = Math.min(minLength, str.length());
        }
        int low = 0, high = minLength;
        while (low < high) {
            int mid = (high - low + 1) / 2 + low;
            if (isCommonPrefix(strs, mid)) {
                low = mid;
            } else {
                high = mid - 1;
            }
        }
        return strs[0].substring(0, low);

    }

    private boolean isCommonPrefix(String[] strs, int length) {
        String str0 = strs[0].substring(0, length);
        int count = strs.length;
        for (int i = 1; i < count; i++) {
            String str = strs[i];
            for (int j = 0; j < length; j++) {
                if (str0.charAt(j) != str.charAt(j)) {
                    return false;
                }
            }
        }
        return true;
    }

    public int maxScoreSightseeingPair(int[] A) {
        int ans = 0, mx = A[0] + 0;
        for (int j = 1; j < A.length; j++) {
            ans = Math.max(ans, mx + A[j] - j);
            mx = Math.max(mx, A[j] + j);
        }
        return ans;
    }

    /**
     * 1480. 一维数组的动态和
     * <p>
     * 给你一个数组 nums 。数组「动态和」的计算公式为：runningSum[i] = sum(nums[0]…nums[i]) 。
     * <p>
     * 请返回 nums 的动态和。
     *
     * @param nums
     * @return
     */
    public int[] runningSum(int[] nums) {
        if (nums == null || nums.length == 0)
            return new int[0];
        int[] runningSum = new int[nums.length];
        runningSum[0] = nums[0];
        int i = 1;
        while (i < nums.length) {
            runningSum[i] = runningSum[i - 1] + nums[i];
            i++;
        }
        return runningSum;
    }

    /**
     * 1470. 重新排列数组
     * <p>
     * 给你一个数组 nums ，数组中有 2n 个元素，按 [x1,x2,...,xn,y1,y2,...,yn] 的格式排列。
     * <p>
     * 请你将数组按 [x1,y1,x2,y2,...,xn,yn] 格式重新排列，返回重排后的数组。
     *
     * @param nums
     * @param n
     * @return
     */
    public int[] shuffle(int[] nums, int n) {
        if (nums == null || nums.length == 0)
            return new int[0];
        if (nums.length < 2 * n)
            n = nums.length / 2;
        int[] result = new int[2 * n];
        for (int i = 0; i < n; i++) {
            result[2 * i] = nums[i];
            result[2 * i + 1] = nums[n + i];
        }
        return result;
    }

    /**
     * 209. 长度最小的子数组
     * <p>
     * 给定一个含有 n 个正整数的数组和一个正整数 s ，找出该数组中满足其和 ≥ s 的长度最小的连续子数组，并返回其长度。如果不存在符合条件的连续子数组，返回 0。
     *
     * @param s
     * @param nums
     * @return
     */
    public int minSubArrayLen(int s, int[] nums) {
        if (nums == null || nums.length == 0)
            return 0;
        int minLength = Integer.MAX_VALUE, sum = 0;
        int left = 0, right = 0;
        while (right < nums.length) {
            sum += nums[right];
            while (sum >= s) {
                minLength = Math.min(minLength, right - left + 1);
                sum -= nums[left];
                left++;
            }
            right++;
        }
        return minLength == Integer.MAX_VALUE ? 0 : minLength;
    }

    /**
     * 215. 数组中的第K个最大元素
     * <p>
     * 在未排序的数组中找到第 k 个最大的元素。请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。
     *
     * @param nums
     * @param k
     * @return
     */
    public int findKthLargest(int[] nums, int k) {
        int heapSize = nums.length;
        buildMaxHeap(nums, heapSize);
        for (int i = nums.length - 1; i >= nums.length - k + 1; i--) {
            swap(nums, 0, i);
            heapSize--;
            maxHeapify(nums, 0, heapSize);
        }
        return nums[0];
    }

    private void buildMaxHeap(int[] a, int heapSize) {
        for (int i = heapSize / 2; i >= 0; i--) {
            maxHeapify(a, i, heapSize);
        }
    }

    private void maxHeapify(int[] a, int i, int heapSize) {
        int l = i * 2 + 1, r = i * 2 + 2, largest = i;
        if (l < heapSize && a[l] > a[largest]) {
            largest = l;
        }
        if (r < heapSize && a[r] > a[largest]) {
            largest = r;
        }
        if (largest != i) {
            swap(a, i, largest);
            maxHeapify(a, largest, heapSize);
        }
    }

    /***
     * 378. 有序矩阵中第K小的元素
     *
     * 给定一个 n x n 矩阵，其中每行和每列元素均按升序排序，找到矩阵中第 k 小的元素。
     * 请注意，它是排序后的第 k 小元素，而不是第 k 个不同的元素。
     *
     * @param matrix
     * @param k
     * @return
     */
    public int kthSmallest(int[][] matrix, int k) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0)
            return 0;
        int n = matrix.length;
        int left = matrix[0][0];
        int right = matrix[n - 1][n - 1];
        while (left < right) {
            int mid = left + ((right - left) >> 1);
            if (checkKthSmallest(matrix, mid, k, n)) {
                right = mid;
            } else
                left = mid + 1;
        }
        return left;

    }

    private boolean checkKthSmallest(int[][] matrix, int mid, int k, int n) {
        int i = n - 1;
        int j = 0;
        int num = 0;
        while (i >= 0 && j < n) {
            if (matrix[i][j] <= mid) {
                num += i + 1;
                j++;
            } else {
                i--;
            }
        }
        return num >= k;
    }

    /**
     * 108. 将有序数组转换为二叉搜索树
     * <p>
     * 将一个按照升序排列的有序数组，转换为一棵高度平衡二叉搜索树。
     * <p>
     * 本题中，一个高度平衡二叉树是指一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过 1。
     *
     * @param nums
     * @return
     */
    public TreeNode sortedArrayToBST(int[] nums) {
        return sortedArrayToBSTHelper(nums, 0, nums.length - 1);
    }

    private TreeNode sortedArrayToBSTHelper(int[] nums, int left, int right) {
        if (left > right)
            return null;
        int mid = (left + right) / 2;
        TreeNode root = new TreeNode(nums[mid]);
        root.left = sortedArrayToBSTHelper(nums, left, mid - 1);
        root.right = sortedArrayToBSTHelper(nums, mid + 1, right);
        return root;
    }

    /**
     * 63. 不同路径 II
     * 一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。
     * <p>
     * 机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。
     * <p>
     * 现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？
     *
     * @param obstacleGrid
     * @return
     */
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        if (obstacleGrid == null || obstacleGrid.length == 0 || obstacleGrid[0].length == 0)
            return 0;
        int width = obstacleGrid.length;
        int hight = obstacleGrid[0].length;
        int[][] result = new int[width][hight];
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < hight; j++) {
                if (obstacleGrid[i][j] == 1) {
                    result[i][j] = 0;
                } else {
                    if (i == 0 && j == 0)
                        result[i][j] = 1;
                    if (i > 0)
                        result[i][j] = result[i - 1][j] + result[i][j];
                    if (j > 0)
                        result[i][j] = result[i][j - 1] + result[i][j];
                }
            }
        }
        return result[width - 1][hight - 1];
    }

    /**
     * 309. 最佳买卖股票时机含冷冻期
     * <p>
     * 给定一个整数数组，其中第 i 个元素代表了第 i 天的股票价格 。​
     * <p>
     * 设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）:
     * <p>
     * 你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
     * 卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。
     *
     * @param prices
     * @return
     */
    public int maxProfit(int[] prices) {
        if (prices == null || prices.length == 0)
            return 0;
        int n = prices.length;
        int f0 = -prices[0];
        int f1 = 0;
        int f2 = 0;
        for (int i = 1; i < n; i++) {
            int newf0 = Math.max(f0, f2 - prices[i]);
            int newf1 = f0 + prices[i];
            int newf2 = Math.max(f1, f2);
            f0 = newf0;
            f1 = newf1;
            f2 = newf2;
        }
        return Math.max(f1, f2);
    }


    /**
     * 两个数组的交集2
     * 给定两个数组，编写一个函数来计算它们的交集。
     *
     * @param nums1
     * @param nums2
     * @return
     */
    public int[] intersect(int[] nums1, int[] nums2) {
        if (nums1 == null || nums1.length == 0 || nums2 == null || nums2.length == 0)
            return new int[0];
        HashMap<Integer, Integer> nums1HM = new HashMap<>();
        for (int num : nums1) {
            int count = nums1HM.getOrDefault(num, 0) + 1;
            nums1HM.put(num, count);
        }
        int[] intersection = new int[Math.min(nums1.length, nums2.length)];
        int index = 0;
        for (int num : nums2) {
            int count = nums1HM.getOrDefault(num, 0);
            if (count > 0) {
                intersection[index++] = num;
                count--;
                if (count > 0)
                    nums1HM.put(num, count);
                else
                    nums1HM.remove(num);
            }

        }
        return Arrays.copyOfRange(intersection, 0, index);
    }

    /**
     * 1512. 好数对的数目
     * <p>
     * 给你一个整数数组 nums 。
     * <p>
     * 如果一组数字 (i,j) 满足 nums[i] == nums[j] 且 i < j ，就可以认为这是一组 好数对 。
     * <p>
     * 返回好数对的数目。
     * <p>
     *  
     *
     * @param nums
     * @return
     */
    public int numIdenticalPairs(int[] nums) {
        if (nums == null || nums.length == 0)
            return 0;
        int count = 0;
        HashMap<Integer, Integer> mHashMap = new HashMap<>();
        for (int i : nums) {
            mHashMap.put(i, mHashMap.getOrDefault(i, -1) + 1);
        }
        for (int i : mHashMap.keySet()) {
            int v = mHashMap.get(i);
            count += (1 + v) * v / 2;
        }
        return count;
    }

    /**
     * 剑指 Offer 11. 旋转数组的最小数字
     *
     * @param numbers
     * @return
     */
    public int minArray(int[] numbers) {
        if (numbers == null || numbers.length == 0) {
            return 0;
        }
        if (numbers.length == 1)
            return numbers[0];
        int low = 0, hight = numbers.length - 1;
        while (low < hight) {
            int pivot = low + (hight - low) / 2;
            if (numbers[pivot] < numbers[hight]) {
                hight = pivot;
            } else if (numbers[pivot] > numbers[hight]) {
                low = pivot + 1;
            } else {
                hight -= 1;
            }
        }
        return numbers[low];

    }

    /**
     * 64. 最小路径和
     * <p>
     * 给定一个包含非负整数的 m x n 网格，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。
     * <p>
     * 说明：每次只能向下或者向右移动一步。
     *
     * @param grid
     * @return
     */
    public int minPathSum(int[][] grid) {
        if (grid == null || grid.length == 0 || grid[0].length == 0)
            return 0;
        int m = grid.length, n = grid[0].length;
        int[][] result = new int[m][n];
        for (int mIndex = 0; mIndex < m; mIndex++) {
            for (int nIndex = 0; nIndex < n; nIndex++) {
                if (mIndex == 0 && nIndex == 0)
                    result[0][0] = grid[0][0];
                else if (mIndex == 0)
                    result[mIndex][nIndex] = result[mIndex][nIndex - 1] + grid[mIndex][nIndex];
                else if (nIndex == 0)
                    result[mIndex][nIndex] = result[mIndex - 1][nIndex] + grid[mIndex][nIndex];
                else {
                    result[mIndex][nIndex] = Math.min(result[mIndex - 1][nIndex], result[mIndex][nIndex - 1]) + grid[mIndex][nIndex];
                }
            }
        }
        return result[m - 1][n - 1];
    }

    /**
     * 410. 分割数组的最大值
     * <p>
     * 给定一个非负整数数组和一个整数 m，你需要将这个数组分成 m 个非空的连续子数组。设计一个算法使得这 m 个子数组各自和的最大值最小。
     * <p>
     * 注意:
     * 数组长度 n 满足以下条件:
     * <p>
     * 1 ≤ n ≤ 1000
     * 1 ≤ m ≤ min(50, n)
     * 示例:
     *
     * @param nums
     * @param m
     * @return
     */
    public int splitArray(int[] nums, int m) {
        int n = nums.length;
        int[][] f = new int[n + 1][m + 1];
        for (int i = 0; i <= n; i++) {
            Arrays.fill(f[i], Integer.MAX_VALUE);
        }
        int[] sub = new int[n + 1];
        for (int i = 0; i < n; i++) {
            sub[i + 1] = sub[i] + nums[i];
        }
        f[0][0] = 0;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= Math.min(i, m); j++) {
                for (int k = 0; k < i; k++) {
                    f[i][j] = Math.min(f[i][j], Math.max(f[k][j - 1], sub[i] - sub[k]));
                }
            }
        }
        return f[n][m];

    }

    /**
     * 329. 矩阵中的最长递增路径
     * <p>
     * 给定一个整数矩阵，找出最长递增路径的长度。
     * <p>
     * 对于每个单元格，你可以往上，下，左，右四个方向移动。 你不能在对角线方向上移动或移动到边界外（即不允许环绕）。
     *
     * @param matrix
     * @return
     */
    public int[][] dirs = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    public int rows, columns;

    public int longestIncreasingPath(int[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0)
            return 0;
        rows = matrix.length;
        columns = matrix[0].length;
        int[][] memo = new int[rows][columns];
        int ans = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                ans = Math.max(ans, lIPDFS(matrix, i, j, memo));
            }
        }
        return ans;
    }

    private int lIPDFS(int[][] matrix, int row, int column, int[][] memo) {
        if (memo[row][column] != 0)
            return memo[row][column];
        memo[row][column]++;
        for (int[] dir : dirs) {
            int newRow = row + dir[0], newColumn = column + dir[1];
            if (newRow >= 0 && newRow < rows && newColumn >= 0 && newColumn < columns && matrix[newRow][newColumn] > matrix[row][column]) {
                memo[row][column] = Math.max(memo[row][column], lIPDFS(matrix, newRow, newColumn, memo) + 1);
            }
        }
        return memo[row][column];
    }

    /**
     * LCP 13. 寻宝
     * 我们得到了一副藏宝图，藏宝图显示，在一个迷宫中存在着未被世人发现的宝藏。
     * <p>
     * 迷宫是一个二维矩阵，用一个字符串数组表示。它标识了唯一的入口（用 'S' 表示），和唯一的宝藏地点（用 'T' 表示）。但是，宝藏被一些隐蔽的机关保护了起来。在地图上有若干个机关点（用 'M' 表示），只有所有机关均被触发，才可以拿到宝藏。
     * <p>
     * 要保持机关的触发，需要把一个重石放在上面。迷宫中有若干个石堆（用 'O' 表示），每个石堆都有无限个足够触发机关的重石。但是由于石头太重，我们一次只能搬一个石头到指定地点。
     * <p>
     * 迷宫中同样有一些墙壁（用 '#' 表示），我们不能走入墙壁。剩余的都是可随意通行的点（用 '.' 表示）。石堆、机关、起点和终点（无论是否能拿到宝藏）也是可以通行的。
     * <p>
     * 我们每步可以选择向上/向下/向左/向右移动一格，并且不能移出迷宫。搬起石头和放下石头不算步数。那么，从起点开始，我们最少需要多少步才能最后拿到宝藏呢？如果无法拿到宝藏，返回 -1 。
     *
     * @param maze
     * @return
     */
    int[] dx = {1, -1, 0, 0};
    int[] dy = {0, 0, 1, -1};
    int n, m;

    public int minimalSteps(String[] maze) {
        n = maze.length;
        m = maze[0].length();

        List<int[]> buttons = new ArrayList<int[]>();
        List<int[]> stones = new ArrayList<>();

        int sx = -1, sy = -1, tx = -1, ty = -1;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (maze[i].charAt(j) == 'M') {
                    buttons.add(new int[]{i, j});
                }
                if (maze[i].charAt(j) == 'O') {
                    stones.add(new int[]{i, j});
                }
                if (maze[i].charAt(j) == 'S') {
                    sx = i;
                    sy = j;
                }
                if (maze[i].charAt(j) == 'T') {
                    tx = i;
                    ty = j;
                }
            }
        }
        int nb = buttons.size();
        int ns = stones.size();
        int[][] startDist = bfsMS(sx, sy, maze);

        if (nb == 0)
            return startDist[tx][ty];
        int[][] dist = new int[nb][nb + 2];
        for (int i = 0; i < nb; i++) {
            Arrays.fill(dist[i], -1);
        }
        int[][][] dd = new int[nb][][];
        for (int i = 0; i < nb; i++) {
            int[][] d = bfsMS(buttons.get(i)[0], buttons.get(i)[1], maze);
            dd[i] = d;
            dist[i][nb + 1] = d[tx][ty];
        }
        for (int i = 0; i < nb; i++) {
            int tmp = -1;
            for (int k = 0; k < ns; k++) {
                int midX = stones.get(k)[0], midY = stones.get(k)[1];
                if (dd[i][midX][midY] != -1 && startDist[midX][midY] != -1) {
                    if (tmp == -1 || tmp > dd[i][midX][midY] + startDist[midX][midY]) {
                        tmp = dd[i][midX][midY] + startDist[midX][midY];
                    }
                }
            }
            dist[i][nb] = tmp;
            for (int j = i + 1; j < nb; j++) {
                int mn = -1;
                for (int k = 0; k < ns; k++) {
                    int midX = stones.get(k)[0], midY = stones.get(k)[1];
                    if (dd[i][midX][midY] != -1 && dd[j][midX][midY] != -1) {
                        if (mn == -1 || mn > dd[i][midX][midY] + dd[j][midX][midY]) {
                            mn = dd[i][midX][midY] + dd[j][midX][midY];
                        }
                    }
                }
                dist[i][j] = mn;
                dist[j][i] = mn;
            }
        }

        // 无法达成的情形
        for (int i = 0; i < nb; i++) {
            if (dist[i][nb] == -1 || dist[i][nb + 1] == -1) {
                return -1;
            }
        }

        // dp 数组， -1 代表没有遍历到
        int[][] dp = new int[1 << nb][nb];
        for (int i = 0; i < 1 << nb; i++) {
            Arrays.fill(dp[i], -1);
        }
        for (int i = 0; i < nb; i++) {
            dp[1 << i][i] = dist[i][nb];
        }

        // 由于更新的状态都比未更新的大，所以直接从小到大遍历即可
        for (int mask = 1; mask < (1 << nb); mask++) {
            for (int i = 0; i < nb; i++) {
                // 当前 dp 是合法的
                if ((mask & (1 << i)) != 0) {
                    for (int j = 0; j < nb; j++) {
                        // j 不在 mask 里
                        if ((mask & (1 << j)) == 0) {
                            int next = mask | (1 << j);
                            if (dp[next][j] == -1 || dp[next][j] > dp[mask][i] + dist[i][j]) {
                                dp[next][j] = dp[mask][i] + dist[i][j];
                            }
                        }
                    }
                }
            }
        }

        int ret = -1;
        int finalMask = (1 << nb) - 1;
        for (int i = 0; i < nb; i++) {
            if (ret == -1 || ret > dp[finalMask][i] + dist[i][nb + 1]) {
                ret = dp[finalMask][i] + dist[i][nb + 1];
            }
        }

        return ret;
    }

    private int[][] bfsMS(int x, int y, String[] maze) {
        int[][] ret = new int[n][m];
        for (int i = 0; i < n; i++) {
            Arrays.fill(ret[i], -1);
        }
        ret[x][y] = 0;
        Queue<int[]> queue = new LinkedList<>();
        queue.offer(new int[]{x, y});
        while (!queue.isEmpty()) {
            int[] p = queue.poll();
            int curx = p[0], cury = p[1];
            for (int k = 0; k < 4; k++) {
                int nx = curx + dx[k], ny = cury + dy[k];
                if (inBoundMS(nx, ny) && maze[nx].charAt(ny) != '#' && ret[nx][ny] == -1) {
                    ret[nx][ny] = ret[curx][cury] + 1;
                    queue.offer(new int[]{nx, ny});
                }
            }
        }
        return ret;
    }

    private boolean inBoundMS(int x, int y) {
        return x >= 0 && x < n && y >= 0 && y < m;
    }

    /**
     * 面试题 08.03. 魔术索引
     * 魔术索引。 在数组A[0...n-1]中，有所谓的魔术索引，满足条件A[i] = i。给定一个有序整数数组，编写一种方法找出魔术索引，若有的话，在数组A中找出一个魔术索引，如果没有，则返回-1。若有多个魔术索引，返回索引值最小的一个。
     *
     * @param nums
     * @return
     */
    public int findMagicIndex(int[] nums) {
        return getAnswer(nums, 0, nums.length - 1);
    }

    private int getAnswer(int[] nums, int left, int right) {
        if (left > right)
            return -1;
        int mid = (right - left) / 2 + left;
        int leftAnswer = getAnswer(nums, left, mid - 1);
        if (leftAnswer != -1)
            return leftAnswer;
        else if (nums[mid] == mid) {
            return mid;
        }
        return getAnswer(nums, mid + 1, right);
    }

    /**
     * 130. 被围绕的区域
     * <p>
     * 给定一个二维的矩阵，包含 'X' 和 'O'（字母 O）。
     * <p>
     * 找到所有被 'X' 围绕的区域，并将这些区域里所有的 'O' 用 'X' 填充。
     *
     * @param board
     */
    private int[] solveDx = {1, -1, 0, 0};
    private int[] solveDy = {0, 0, 1, -1};

    public void solve(char[][] board) {
        int n = board.length;
        if (n == 0)
            return;
        int m = board[0].length;
        Queue<int[]> queue = new LinkedList<>();
        for (int i = 0; i < n; i++) {
            if (board[i][0] == 'O')
                queue.offer(new int[]{i, 0});
            if (board[i][m - 1] == 'O')
                queue.offer(new int[]{i, m - 1});

        }
        for (int i = 1; i < m - 1; i++) {
            if (board[0][i] == 'O')
                queue.offer(new int[]{0, i});
            if (board[n - 1][i] == 'O')
                queue.offer(new int[]{n - 1, i});
        }
        while (!queue.isEmpty()) {
            int[] cell = queue.poll();
            int x = cell[0], y = cell[1];
            board[x][y] = 'A';
            for (int i = 0; i < 4; i++) {
                int mx = x + solveDx[i], my = y + solveDy[i];
                if (mx < 0 || my < 0 || mx >= n || my >= m || board[mx][my] != 'O')
                    continue;
                queue.offer(new int[]{mx, my});
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (board[i][j] == 'A')
                    board[i][j] = 'O';
                else if (board[i][j] == 'O') {
                    board[i][j] = 'X';
                }
            }
        }
    }

    /**
     * 733. 图像渲染
     * <p>
     * 有一幅以二维整数数组表示的图画，每一个整数表示该图画的像素值大小，数值在 0 到 65535 之间。
     * <p>
     * 给你一个坐标 (sr, sc) 表示图像渲染开始的像素值（行 ，列）和一个新的颜色值 newColor，让你重新上色这幅图像。
     * <p>
     * 为了完成上色工作，从初始坐标开始，记录初始坐标的上下左右四个方向上像素值与初始坐标相同的相连像素点，接着再记录这四个方向上符合条件的像素点与他们对应四个方向上像素值与初始坐标相同的相连像素点，……，重复该过程。将所有有记录的像素点的颜色值改为新的颜色值。
     * <p>
     * 最后返回经过上色渲染后的图像
     *
     * @param image
     * @param sr
     * @param sc
     * @param newColor
     * @return
     */
    int[] dxFloodFill = {1, 0, 0, -1};
    int[] dyFloodFill = {0, 1, -1, 0};

    public int[][] floodFill(int[][] image, int sr, int sc, int newColor) {
        int currColor = image[sr][sc];
        if (currColor == newColor)
            return image;
        int n = image.length, m = image[0].length;
        Queue<int[]> queue = new LinkedList<>();
        queue.offer(new int[]{sr, sc});
        image[sr][sc] = newColor;
        while (!queue.isEmpty()) {
            int[] cell = queue.poll();
            int x = cell[0], y = cell[1];
            for (int i = 0; i < 4; i++) {
                int mx = x + dxFloodFill[i], my = y + dyFloodFill[i];
                if (mx >= 0 && mx < n && my >= 0 && my < m && image[mx][my] == currColor) {
                    queue.offer(new int[]{mx, my});
                    image[mx][my] = newColor;
                }
            }
        }
        return image;
    }

    /**
     * 491. 递增子序列
     * <p>
     * 给定一个整型数组, 你的任务是找到所有该数组的递增子序列，递增子序列的长度至少是2。
     *
     * @param nums
     * @return
     */
    List<Integer> temp = new ArrayList<>();
    List<List<Integer>> ans = new ArrayList<>();

    public List<List<Integer>> findSubsequences(int[] nums) {
        dfsFindSubsequences(0, Integer.MIN_VALUE, nums);
        return ans;
    }

    private void dfsFindSubsequences(int cur, int last, int[] nums) {
        if (cur == nums.length) {
            if (temp.size() >= 2) {
                ans.add(new ArrayList<>(temp));
            }
            return;
        }
        if (nums[cur] >= last) {
            temp.add(nums[cur]);
            dfsFindSubsequences(cur + 1, nums[cur], nums);
            temp.remove(temp.size() - 1);
        }
        if (nums[cur] != last) {
            dfsFindSubsequences(cur + 1, last, nums);
        }
    }

    /**
     * 39. 组合总和
     * <p>
     * 给定一个无重复元素的数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。
     * candidates 中的数字可以无限制重复被选取。
     *
     * @param candidates
     * @param target
     * @return
     */
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> ans = new ArrayList<>();
        List<Integer> combine = new ArrayList<>();
        dfsCombinationSum(candidates, target, ans, combine, 0);
        return ans;
    }

    private void dfsCombinationSum(int[] candidates, int target, List<List<Integer>> ans, List<Integer> combine, int idx) {
        if (idx == candidates.length) {
            return;
        }
        if (target == 0) {
            ans.add(new ArrayList<>(combine));
            return;
        }
        dfsCombinationSum(candidates, target, ans, combine, idx + 1);
        if (target - candidates[idx] >= 0) {
            combine.add(candidates[idx]);
            dfsCombinationSum(candidates, target - candidates[idx], ans, combine, idx);
            combine.remove(combine.size() - 1);
        }
    }

    /**
     * 40. 组合总和 II
     * 给定一个数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。
     * <p>
     * candidates 中的每个数字在每个组合中只能使用一次。
     * <p>
     * 说明：
     * <p>
     * 所有数字（包括目标数）都是正整数。
     * 解集不能包含重复的组合。 
     *
     * @param candidates
     * @param target
     * @return
     */
    List<int[]> freq = new ArrayList<>();
    List<List<Integer>> ansCombinationSum2 = new ArrayList<>();
    List<Integer> sequence = new ArrayList<>();

    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        Arrays.sort(candidates);
        for (int num : candidates) {
            int size = freq.size();
            if (freq.isEmpty() || num != freq.get(size - 1)[0]) {
                freq.add(new int[]{num, 1});
            } else {
                freq.get(size - 1)[1]++;
            }
        }
        dfsCombinationSum2(0, target);
        return ansCombinationSum2;
    }

    private void dfsCombinationSum2(int pos, int rest) {
        if (rest == 0) {
            ansCombinationSum2.add(new ArrayList<>(sequence));
            return;
        }
        if (pos == freq.size() || rest < freq.get(pos)[0]) {
            return;
        }
        dfsCombinationSum2(pos + 1, rest);
        int most = Math.min(rest / freq.get(pos)[0], freq.get(pos)[1]);
        for (int i = 1; i <= most; i++) {
            sequence.add(freq.get(pos)[0]);
            dfsCombinationSum2(pos + 1, rest - i * freq.get(pos)[0]);
        }
        for (int i = 1; i <= most; i++) {
            sequence.remove(sequence.size() - 1);
        }
    }

    /**
     * 79. 单词搜索
     * 给定一个二维网格和一个单词，找出该单词是否存在于网格中。
     * 单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。
     *
     * @param board
     * @param word
     * @return
     */
    public boolean exist(char[][] board, String word) {
        if (board == null || board.length == 0)
            return false;
        int h = board.length, w = board[0].length;
        boolean[][] visited = new boolean[h][w];
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                boolean flag = checkExist(board, visited, i, j, word, 0);
                if (flag) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean checkExist(char[][] board, boolean[][] visited, int i, int j, String s, int k) {
        if (board[i][j] != s.charAt(k)) {
            return false;
        } else if (k == s.length() - 1) {
            return true;
        }
        visited[i][j] = true;
        int[][] directions = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        boolean result = false;
        for (int[] dir : directions) {
            int newi = i + dir[0], newj = j + dir[1];
            if (newi >= 0 && newi < board.length && newj >= 0 && newj < board[0].length) {
                if (!visited[newi][newj]) {
                    boolean flag = checkExist(board, visited, newi, newj, s, k + 1);
                    if (flag) {
                        result = true;
                        break;
                    }
                }
            }
        }
        visited[i][j] = false;
        return result;

    }

    /**
     * 78. 子集
     * 给定一组不含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。
     * <p>
     * 说明：解集不能包含重复的子集。
     *
     * @param nums
     * @return
     */
    List<Integer> tSubsets = new ArrayList<>();
    List<List<Integer>> ansSubsets = new ArrayList<>();

    public List<List<Integer>> subsets(int[] nums) {
        int n = nums.length;
        for (int mask = 0; mask < (1 << n); mask++) {
            tSubsets.clear();
            for (int i = 0; i < n; i++) {

                if ((mask & (1 << i)) != 0) {
                    tSubsets.add(nums[i]);
                }
            }
            ansSubsets.add(new ArrayList<>(tSubsets));
        }
        return ansSubsets;
    }

    /**
     * 416. 分割等和子集
     * 给定一个只包含正整数的非空数组。是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。
     * <p>
     * 注意:
     * <p>
     * 每个数组中的元素不会超过 100
     * 数组的大小不会超过 200
     *
     * @param nums
     * @return
     */
    public boolean canPartition(int[] nums) {
        int n = nums.length;
        if (n < 2) {
            return false;
        }
        int sum = 0, maxNum = 0;
        for (int num : nums) {
            sum += num;
            maxNum = Math.max(maxNum, num);
        }
        if (sum % 2 != 0) {
            return false;
        }
        int target = sum / 2;
        if (maxNum > target) {
            return false;
        }
        boolean[][] dp = new boolean[n][target + 1];
        for (int i = 0; i < n; i++) {
            dp[i][0] = true;
        }
        dp[0][nums[0]] = true;
        for (int i = 1; i < n; i++) {
            dp[i][0] = true;
        }
        dp[0][nums[0]] = true;
        for (int i = 1; i < n; i++) {
            int num = nums[i];
            for (int j = 1; j <= target; j++) {
                if (j >= num) {
                    dp[i][j] = dp[i - 1][j] | dp[i - 1][j - num];
                } else {
                    dp[i][j] = dp[i - 1][j];
                }
            }
        }
        return dp[n - 1][target];
    }

    /**
     * 977. 有序数组的平方
     *
     * @param A
     * @return
     */
    public int[] sortedSquares(int[] A) {
        if (A == null) {
            return null;
        }
        int[] result = new int[A.length];
        for (int i = 0; i < A.length; i++) {
            result[i] = A[i] * A[i];
        }
        Arrays.sort(result);
        return result;
    }

    public int[] sortedSquares_v2(int[] A) {
        int n = A.length;
        int[] ans = new int[n];
        for (int i = 0, j = n - 1, pos = n - 1; i <= j; ) {
            if (A[i] * A[i] > A[j] * A[j]) {
                ans[pos] = A[i] * A[i];
                i++;
            } else {
                ans[pos] = A[j] * A[j];
                j--;
            }
            pos--;

        }
        return ans;
    }

    /**
     * 845. 数组中的最长山脉
     * 我们把数组 A 中符合下列属性的任意连续子数组 B 称为 “山脉”：
     * <p>
     * B.length >= 3
     * 存在 0 < i < B.length - 1 使得 B[0] < B[1] < ... B[i-1] < B[i] > B[i+1] > ... > B[B.length - 1]
     * （注意：B 可以是 A 的任意子数组，包括整个数组 A。）
     * <p>
     * 给出一个整数数组 A，返回最长 “山脉” 的长度。
     * <p>
     * 如果不含有 “山脉” 则返回 0。
     *
     * @param A
     * @return
     */
    public int longestMountain(int[] A) {
        int result = 0;
        int n = A.length;
        if (A != null && n > 2) {
            int left = 0;
            while (left + 2 < n) {
                int right = left + 1;
                if (A[left] < A[left + 1]) {
                    while (right + 1 < n && A[right] < A[right + 1]) {
                        right++;
                    }
                    if (right < n - 1 && A[right] > A[right + 1]) {
                        while (right + 1 < n && A[right] > A[right + 1]) {
                            right++;
                        }
                        result = Math.max(result, right - left + 1);
                    } else {
                        right++;
                    }
                }
                left = right;
            }
        }
        return result;
    }

    /**
     * 1207. 独一无二的出现次数
     * <p>
     * 给你一个整数数组 arr，请你帮忙统计数组中每个数的出现次数。
     * <p>
     * 如果每个数的出现次数都是独一无二的，就返回 true；否则返回 false。
     *
     * @param arr
     * @return
     */
    public boolean uniqueOccurrences(int[] arr) {
        Map<Integer, Integer> occur = new HashMap<>();
        for (int x : arr) {
            occur.put(x, occur.getOrDefault(x, 0) + 1);
        }
        Set<Integer> times = new HashSet<>();
        for (Map.Entry<Integer, Integer> x : occur.entrySet()) {
            times.add(x.getValue());
        }
        return times.size() == occur.size();
    }

    /**
     * 349. 两个数组的交集
     * <p>
     * 给定两个数组，编写一个函数来计算它们的交集。
     *
     * @param nums1
     * @param nums2
     * @return
     */
    public int[] intersection(int[] nums1, int[] nums2) {
        if (nums1 == null || nums2 == null || nums1.length == 0 || nums2.length == 0) {
            return new int[0];
        }
        Set<Integer> numCount = new HashSet<>();
        for (int n : nums1) {
            numCount.add(n);
        }
        List<Integer> list = new ArrayList<>();
        for (int n : nums2) {
            if (numCount.contains(n)) {
                list.add(n);
                numCount.remove(n);
            }
        }
        int[] result = new int[list.size()];
        for (int i = 0; i < list.size(); i++) {
            result[i] = list.get(i);
        }
        return result;
    }

    /**
     * 941. 有效的山脉数组
     * <p>
     * 给定一个整数数组 A，如果它是有效的山脉数组就返回 true，否则返回 false。
     * <p>
     * 让我们回顾一下，如果 A 满足下述条件，那么它是一个山脉数组：
     * <p>
     * A.length >= 3
     * 在 0 < i < A.length - 1 条件下，存在 i 使得：
     * A[0] < A[1] < ... A[i-1] < A[i]
     * A[i] > A[i+1] > ... > A[A.length - 1]
     *
     * @param A
     * @return
     */
    public boolean validMountainArray(int[] A) {
        int N = A.length;
        int i = 0;
        while (i + 1 < N && A[i] < A[i + 1]) {
            i++;
        }
        if (i == 0 || i == N - 1) {
            return false;
        }
        while (i + 1 < N && A[i] > A[i + 1]) {
            i++;
        }
        return i == N - 1;
    }

    /**
     * 57. 插入区间
     * 给出一个无重叠的 ，按照区间起始端点排序的区间列表。
     * <p>
     * 在列表中插入一个新的区间，你需要确保列表中的区间仍然有序且不重叠（如果有必要的话，可以合并区间）。
     *
     * @param intervals
     * @param newInterval
     * @return
     */
    public int[][] insert(int[][] intervals, int[] newInterval) {
        if (intervals == null && newInterval == null) {
            return null;
        }
        int left = newInterval[0];
        int right = newInterval[1];
        boolean placed = false;
        List<int[]> ansList = new ArrayList<>();
        for (int[] interval : intervals) {
            if (interval[0] > right) {
                if (!placed) {
                    ansList.add(new int[]{left, right});
                    placed = true;
                }
                ansList.add(interval);
            } else if (interval[1] < left) {
                ansList.add(interval);
            } else {
                left = Math.min(left, interval[0]);
                right = Math.max(right, interval[1]);
            }
        }
        if (!placed) {
            ansList.add(new int[]{left, right});
        }
        int[][] ans = new int[ansList.size()][2];
        for (int i = 0; i < ansList.size(); i++) {
            ans[i] = ansList.get(i);
        }
        return ans;
    }

    /**
     * 1356. 根据数字二进制下 1 的数目排序
     * <p>
     * 给你一个整数数组 arr 。请你将数组中的元素按照其二进制表示中数字 1 的数目升序排序。
     * <p>
     * 如果存在多个数字二进制中 1 的数目相同，则必须将它们按照数值大小升序排列。
     * <p>
     * 请你返回排序后的数组。
     *
     * @param arr
     * @return
     */
    public int[] sortByBits(int[] arr) {
        int[] bit = new int[10001];
        List<Integer> list = new ArrayList<>();
        int count = 0, tmp = 0;
        for (int x : arr) {
            list.add(x);
            count = 0;
            tmp = x;
            while (tmp != 0) {
                count += tmp % 2;
                tmp /= 2;
            }
            bit[x] = count;
        }
        Collections.sort(list, new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                if (bit[o1] != bit[o2]) {
                    return bit[o1] - bit[o2];
                } else {
                    return o1 - o2;
                }
            }
        });
        for (int i = 0; i < arr.length; i++) {
            arr[i] = list.get(i);
        }
        return arr;
    }

    /**
     * 给定一个整数数组nums，返回区间和在[lower, upper]之间的个数，包含lower和upper。
     * 区间和S(i, j)表示在nums中，位置从i到j的元素之和，包含i和j(i ≤ j)。
     * <p>
     * 说明:
     * 最直观的算法复杂度是O(n2) ，请在此基础上优化你的算法。
     *
     * @param nums
     * @param lower
     * @param upper
     * @return
     */
    public int countRangeSum(int[] nums, int lower, int upper) {
        long s = 0;
        long[] sum = new long[nums.length + 1];
        for (int i = 0; i < nums.length; i++) {
            s += nums[i];
            sum[i + 1] = s;
        }
        return countRangeSumRecursive(sum, lower, upper, 0, sum.length - 1);
    }

    private int countRangeSumRecursive(long[] sum, int lower, int upper, int left, int right) {
        if (left == right) {
            return 0;
        } else {
            int mid = (left + right) / 2;
            int n1 = countRangeSumRecursive(sum, lower, upper, left, mid);
            int n2 = countRangeSumRecursive(sum, lower, upper, mid + 1, right);
            int ret = n1 + n2;

            int i = left;
            int l = mid + 1;
            int r = mid + 1;
            while (i <= mid){
                while (l <= right && sum[l] - sum[i] < lower){
                    l++;
                }
                while (r <= right && sum[r] - sum[i] <= upper){
                    r++;
                }
                ret += r - l;
                i++;
            }
            // 随后合并两个排序数组
            int[] sorted = new int[right - left + 1];
            int p1 = left, p2 = mid + 1;
            int p = 0;
            while (p1 <= mid || p2 <= right) {
                if (p1 > mid) {
                    sorted[p++] = (int) sum[p2++];
                } else if (p2 > right) {
                    sorted[p++] = (int) sum[p1++];
                } else {
                    if (sum[p1] < sum[p2]) {
                        sorted[p++] = (int) sum[p1++];
                    } else {
                        sorted[p++] = (int) sum[p2++];
                    }
                }
            }
            for (int j = 0; j < sorted.length; j++) {
                sum[left + j] = sorted[j];
            }
            return ret;
        }
    }

    /**
     *
     * @param nums
     */
    public void nextPermutation(int[] nums) {
        int i = nums.length - 2;
        while (i >= 0 && nums[i] >= nums[i + 1]){
            i--;
        }
        if(i >= 0){
            int j = nums.length - 1;
            while (j >= 0 && nums[i] >= nums[j]){
                j--;
            }
            swapNextPermutation(nums,i,j);
        }
        reverseNextPermutation(nums,i + 1);
    }

    private void swapNextPermutation(int[] nums,int i, int j){
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    private void reverseNextPermutation(int[] nums,int start){
        int left = start, right = nums.length - 1;
        while (left < right){
            swapNextPermutation(nums,left,right);
            left++;
            right--;
        }
    }

    /**
     * 922. 按奇偶排序数组 II
     *
     * 给定一个非负整数数组 A， A 中一半整数是奇数，一半整数是偶数。
     *
     * 对数组进行排序，以便当 A[i] 为奇数时，i 也是奇数；当 A[i] 为偶数时， i 也是偶数。
     *
     * 你可以返回任何满足上述条件的数组作为答案。
     *
     *
     * @param A
     * @return
     */
    public int[] sortArrayByParityII(int[] A) {
        if(A == null || A.length < 2){
            return A;
        }
        int i = 0, j = 1;
        while (i < A.length && j < A.length){
            if(A[i] % 2 != 0){
                while (j < A.length){
                    if(A[j] % 2 == 0){
                        int temp = A[j];
                        A[j] = A[i];
                        A[i] = temp;
                        j += 2;
                        break;
                    }
                    j+=2;
                }
            }
            i += 2;
        }
        return A;
    }
}
