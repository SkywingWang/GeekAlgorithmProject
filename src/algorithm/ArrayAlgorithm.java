package algorithm;

import data.MountainArray;

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
        if(grid == null || grid.length == 0)
            return 0;
        int nr = grid.length;
        int nc = grid[0].length;
        int num_islands = 0;
        for(int r = 0; r < nr; r++){
            for(int c = 0; c < nc; c++){
                if(grid[r][c] == '1'){
                    ++ num_islands;
                    numIslandsDFS(grid,r,c);
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
     * @param nums
     * @param k
     * @return
     */
    public int numberOfSubarrays(int[] nums, int k) {
        if(nums == null || nums.length == 0 || nums.length < k) return 0;
        int left = 0,right = 0;
        int count = 0;
        int res = 0;
        int preEven = 0;
        while (right < nums.length){
            if(count < k){
                if(nums[right] % 2 != 0) count ++;
                right ++;
            }
            if(count == k){
                preEven = 0;
                while (count == k){
                    res++;
                    if(nums[left] % 2 != 0) count--;
                    left++;
                    preEven++;
                }
            }else res += preEven;
        }
        return res;
    }

    /**
     * 46. 全排列
     *
     * 给定一个 没有重复 数字的序列，返回其所有可能的全排列。
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> permute(int[] nums) {
        if(nums == null || nums.length == 0)
            return null;
        List<List<Integer>> res = new LinkedList<>();
        ArrayList<Integer> output = new ArrayList<Integer>();
        for(int num:nums)
            output.add(num);
        int n = nums.length;
        backtrack(n,output,res,0);
        return res;
    }

    private void backtrack(int n, ArrayList<Integer> output,List<List<Integer>> res,int first){
        if(first == n)
            res.add(new ArrayList<Integer>(output));
        for(int i = first; i < n; i++){
            Collections.swap(output,first,i);
            backtrack(n,output,res,first + 1);
            Collections.swap(output,first,i);
        }
    }

    /**
     * 33. 搜索旋转排序数组
     *
     *
     * 假设按照升序排序的数组在预先未知的某个点上进行了旋转。

     * ( 例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] )。

     搜索一个给定的目标值，如果数组中存在这个目标值，则返回它的索引，否则返回 -1 。

     你可以假设数组中不存在重复的元素。

     你的算法时间复杂度必须是 O(log n) 级别。
     *
     * @param nums
     * @param target
     * @return
     */
    public int search(int[] nums, int target) {
        if(nums == null || nums.length == 0)
            return -1;
        int length = nums.length;
        if(length == 1)
            return nums[0] == target ? 0 : -1;
        int left = 0, right = length - 1;
        while (left <= right){
            int mid = (left + right) / 2;
            if(nums[mid] == target) return mid;
            if(nums[0] <= nums[mid]){
                if(nums[0] <= target && target < nums[mid]){
                    right = mid - 1;
                }else{
                    left = mid + 1;
                }
            }else{
                if(nums[mid] < target && target <= nums[length - 1])
                    left = mid + 1;
                else
                    right = mid - 1;
            }
        }
        return -1;
    }


    /**
     * 面试题56 - I. 数组中数字出现的次数
     *
     * 一个整型数组 nums 里除两个数字之外，其他数字都出现了两次。请写程序找出这两个只出现一次的数字。要求时间复杂度是O(n)，空间复杂度是O(1)。
     *
     * @param nums
     * @return
     */
    public int[] singleNumbers(int[] nums) {
        int sum = 0;
        for(int i = 0; i < nums.length; i++){
            sum ^= nums[i];
        }
        int first = 1;
        while ((sum & first) == 0)
            first = first << 1;
        int[] result = new int[2];
        for(int i=0;i<nums.length;i++){
            //将数组分类。
            if((nums[i]&first)==0){
                result[0]^=nums[i];
            }
            else{
                result[1]^=nums[i];
            }
        }
        return result;
    }

    /**
     * 1095. 山脉数组中查找目标值
     * （这是一个 交互式问题 ）
     *
     * 给你一个 山脉数组 mountainArr，请你返回能够使得 mountainArr.get(index) 等于 target 最小 的下标 index 值。
     *
     * 如果不存在这样的下标 index，就请返回 -1。
     *
     *  
     *
     * 何为山脉数组？如果数组 A 是一个山脉数组的话，那它满足如下条件：
     *
     * 首先，A.length >= 3
     *
     * 其次，在 0 < i < A.length - 1 条件下，存在 i 使得：
     *
     * A[0] < A[1] < ... A[i-1] < A[i]
     * A[i] > A[i+1] > ... > A[A.length - 1]
     *  
     *
     * 你将 不能直接访问该山脉数组，必须通过 MountainArray 接口来获取数据：
     *
     * MountainArray.get(k) - 会返回数组中索引为k 的元素（下标从 0 开始）
     * MountainArray.length() - 会返回该数组的长度
     *  
     *
     * 注意：
     *
     * 对 MountainArray.get 发起超过 100 次调用的提交将被视为错误答案。此外，任何试图规避判题系统的解决方案都将会导致比赛资格被取消。
     *
     * 为了帮助大家更好地理解交互式问题，我们准备了一个样例 “答案”：https://leetcode-cn.com/playground/RKhe3ave，请注意这 不是一个正确答案。
     *
     * @param target
     * @param mountainArr
     * @return
     */
    public int findInMountainArray(int target, MountainArray mountainArr) {
        if(mountainArr == null || mountainArr.length() == 0)
            return -1;
        int left = 0, right = mountainArr.length() - 1;
        int mid =  left + ((right - left) >> 1);
        while (left < right){
            if(mountainArr.get(mid) < mountainArr.get(mid + 1))
                left = mid + 1;
            else
                right = mid;
            mid = left + ((right - left) >> 1);
        }
        left = 0;
        int tmp = right;
        int midLeft = (left + right) / 2;
        while (left < right){
            if(mountainArr.get(midLeft) == target) return midLeft;
            if(mountainArr.get(midLeft) > target)
                right = midLeft;
            else
                left = midLeft + 1;
            midLeft = (left + right) / 2;
        }
        left = tmp;
        right = mountainArr.length();
        int midRight = (left + right) / 2;
        while (left < right){
            if(mountainArr.get(midRight) == target) return midRight;
            if(mountainArr.get(midRight) > target)
                left = midRight + 1;
            else
                right = midRight;
            midRight = (left + right) /2;
        }
        return -1;
    }

    /**
     * 1313. 解压缩编码列表
     *
     * 给你一个以行程长度编码压缩的整数列表 nums 。
     *
     * 考虑每对相邻的两个元素 [freq, val] = [nums[2*i], nums[2*i+1]] （其中 i >= 0 ），每一对都表示解压后子列表中有 freq 个值为 val 的元素，你需要从左到右连接所有子列表以生成解压后的列表。
     *
     * 请你返回解压后的列表。
     *
     * @param nums
     * @return
     */
    public int[] decompressRLElist(int[] nums) {
        if(nums == null || nums.length < 2)
            return new int[0];
        ArrayList<Integer> list = new ArrayList<>();
        int i = 0;
        while (2 * i < nums.length){
            for(int n = 0; n < nums[2 * i]; n++){
                list.add(nums[2 * i + 1]);
            }
            i ++;
        }
        int[] resultArray = new int[list.size()];
        for(int n = 0; n < list.size();n++){
            resultArray[n] = list.get(n);
        }
        return resultArray;
    }

    /**
     * 1365. 有多少小于当前数字的数字
     *
     * 给你一个数组 nums，对于其中每个元素 nums[i]，请你统计数组中比它小的所有数字的数目。
     *
     * 换而言之，对于每个 nums[i] 你必须计算出有效的 j 的数量，其中 j 满足 j != i 且 nums[j] < nums[i] 。
     *
     * 以数组形式返回答案。
     *
     * @param nums
     * @return
     */
    public int[] smallerNumbersThanCurrent(int[] nums) {
        if(nums == null || nums.length == 0)
            return new int[0];
        int[] temp = new int[101];
        for(int i = 0; i < nums.length;i++){
            temp[nums[i]] += 1;
        }
        for(int i = 1; i < temp.length; i++){
            temp[i] = temp[i - 1] + temp[i];
        }
        int[] result = new int[nums.length];
        for(int i = 0; i < result.length; i ++){
            if(nums[i]==0)
                result[i] = 0;
            else{
                result[i] = temp[nums[i] - 1];
            }
        }
        return result;
    }

    /**
     * 3. 无重复字符的最长子串
     * 给定一个字符串，请你找出其中不含有重复字符的 最长子串 的长度。
     * @param s
     * @return
     */
    public int lengthOfLongestSubstring(String s) {
        Set<Character> occ = new HashSet<>();
        int n = s.length();
        int rk = -1,ans = 0;
        for(int i = 0; i < n; ++i){
            if(i!=0)
                occ.remove(s.charAt(i - 1));
            while (rk + 1 < n && !occ.contains(s.charAt(rk + 1))){
                occ.add(s.charAt(rk + 1));
                ++rk;
            }
            ans = Math.max(ans,rk - i + 1);
        }
        return ans;

    }

    /**
     * 1389. 按既定顺序创建目标数组
     * 给你两个整数数组 nums 和 index。你需要按照以下规则创建目标数组：
     *
     * 目标数组 target 最初为空。
     * 按从左到右的顺序依次读取 nums[i] 和 index[i]，在 target 数组中的下标 index[i] 处插入值 nums[i] 。
     * 重复上一步，直到在 nums 和 index 中都没有要读取的元素。
     * 请你返回目标数组。
     *
     * 题目保证数字插入位置总是存在。
     *
     * @param nums
     * @param index
     * @return
     */
    public int[] createTargetArray(int[] nums, int[] index) {
        if(nums == null || nums.length == 0 || index == null || index.length == 0)
            return new int[0];
        ArrayList<Integer> resultList = new ArrayList<>();
        for(int i = 0; i < index.length && i < nums.length; i++){
            resultList.add(index[i],nums[i]);
        }
        int[] result = new int[resultList.size()];
        for(int i = 0; i < resultList.size();i++){
            result[i] = resultList.get(i);
        }
        return result;
    }

    /**
     * 45. 跳跃游戏 II
     * 给定一个非负整数数组，你最初位于数组的第一个位置。
     *
     * 数组中的每个元素代表你在该位置可以跳跃的最大长度。
     *
     * 你的目标是使用最少的跳跃次数到达数组的最后一个位置。
     *
     * @param nums
     * @return
     */
    public int jump(int[] nums) {
        if(nums == null || nums.length == 0)
            return 0;
        int position = nums.length - 1;
        int step = 0;
        while (position > 0){
            for(int i = 0 ; i < position; i++){
                if(i + nums[i] >= position){
                    position = i;
                    step++;
                    break;
                }
            }
        }
        return step;
    }

    public int jump2(int[] nums){
        if(nums == null || nums.length == 0)
            return 0;
        int maxPosition = 0;
        int step = 0;
        int end = 0;
        for(int i = 0; i < nums.length; i++){
            maxPosition = Math.max(maxPosition,i + nums[i]);
            if(end == i){
                end = maxPosition;
                step ++;
            }
        }
        return step;
    }
}
