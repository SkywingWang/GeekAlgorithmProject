package algorithm;

/**
 * created by Sven
 * on 2019-12-02
 *
 * 搜索插入位置
 *
 */

public class SearchInsertArray {
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
}
