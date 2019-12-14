package algorithm;

import java.util.HashMap;
import java.util.Map;

/**
 * created by Sven
 * on 2019-12-14
 *
 * 数组Array的相关算法
 *
 */
public class ArrayAlgorithm {

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

    /**
     * created by Sven
     * on 2019-12-09
     *
     * 数组中的连续子数组的最大和
     *
     */
    public int maxSubArray(int[] nums) {
        int ans = nums[0];
        int sum = 0;
        for(int num: nums) {
            if(sum > 0) {
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
     *
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
     *
     * 删除数组重复元素
     */
    public int removeDuplicates(int[] nums){
        if(nums == null)
            return 0;
        if(nums.length < 2)
            return nums.length;
        int k = 0;
        for(int i = 1; i<nums.length;i++){
            if(nums[i] != nums[k]){
                k++;
                nums[k] = nums[i];
            }
        }
        return k + 1;

    }

    /**
     * created by Sven
     * on 2019-11-29
     *
     * 数组右移N步
     */
    public void rotate(int[] nums, int k) {
        if (nums == null || nums.length == 0 || k == 0)
            return;
        k = k % nums.length;
        reverse(nums,0,nums.length - 1);
        reverse(nums,0,k-1);
        reverse(nums,k,nums.length - 1);
        return;
    }

    private void reverse(int[] nums,int left,int right){
        while (left < right)
            swap(nums,left++,right--);

    }

    private void swap(int[] nums,int i,int j){
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    public void rotateMy(int[] nums,int k){
        if (nums == null || nums.length == 0 || k == 0)
            return;
        k = k % nums.length;
        for(int i = 0; i < nums.length / 2; i++){
            swap(nums,i,nums.length - 1 - i);
        }
        for(int i = 0; i < k / 2; i++){
            swap(nums,i,k - 1 - i);
        }
        for(int i = k; i < (nums.length + k) / 2; i++){
            swap(nums,i,nums.length + k - i - 1);
        }
    }

    /**
     * created by Sven
     * on 2019-12-02
     *
     * 搜索插入位置
     *
     */
    public int searchInsert(int[] nums, int target) {
        if(nums == null || nums.length == 0){
            nums = new int[]{target};
            return 0;
        }
        int left = 0;
        int right = nums.length -1;
        int k = 0;
        while(left < right){
            k = (left + right + 1)/2;
            if(nums[k] < target)
                left = k;
            else if(nums[k] > target)
                right = k - 1;
            else
                return k;
        }
        if(target <= nums[k]){
            if(k == 0)
                return 0;
            else if(target > nums[k - 1])
                return k;
            else
                return k - 1;
        }else
            return k + 1;
    }


    /**
     * created by Sven
     * on 2019-12-03
     *
     * 两数之和
     *
     * 给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标。
     *
     * 你可以假设每种输入只会对应一个答案。但是，你不能重复利用这个数组中同样的元素。
     *
     */
    public int[] twoSum(int[] nums, int target) {
        int[] result = new int[2];
        if(nums == null || nums.length < 2)
            return result;
        for(int i = 0; i < nums.length - 1; i++){
            for(int j = i + 1;j<nums.length;j++){
                if(nums[i] + nums[j] == target){
                    result[0] = i;
                    result[1] = j;
                    return result;
                }

            }
        }
        return result;

    }

    public int[] twoSum_2(int[] nums,int target){
        int[] result = new int[2];
        Map<Integer,Integer> numMap = new HashMap<Integer, Integer>();
        for(int i = 0; i < nums.length;i++){
            numMap.put(nums[i],i);
        }
        for(int i = 0; i < nums.length;i++){
            int v = target - nums[i];
            if(numMap.containsKey(v) && i != numMap.get(v)){
                result[0] = i;
                result[1] = numMap.get(v);
                return result;
            }
        }
        return result;
    }
}
