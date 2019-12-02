package algorithm;

public class MergeOrderedArray {
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        if(nums1 == null || nums2 == null || nums2.length == 0)
            return;
        int tmp = 0;
        for(int i = 0;i < m;i++){
            if(nums1[i] > nums2[0]){
                tmp = nums1[i];
                nums1[i] = nums2[0];
                boolean isChange = false;
                for(int j = 0; j < n - 1;j++){
                    if(nums2[j + 1] < tmp){
                        nums2[j] = nums2[j + 1];
                    }else{
                        nums2[j] = tmp;
                        isChange = true;
                        break;
                    }
                }
                if(!isChange)
                    nums2[n-1] = tmp;
            }
        }
        for(int i = 0; i < n; i++){
            nums1[m + i] = nums2[i];
        }
    }
}
