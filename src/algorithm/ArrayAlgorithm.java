package algorithm;

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
}
