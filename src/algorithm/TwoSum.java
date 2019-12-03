package algorithm;

import java.util.HashMap;
import java.util.Map;

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
public class TwoSum {
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
