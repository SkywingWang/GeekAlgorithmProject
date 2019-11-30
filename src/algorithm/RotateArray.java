package algorithm;

/**
 * created by Sven
 * on 2019-11-29
 *
 * 数组右移N步
 */

public class RotateArray {
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
}
