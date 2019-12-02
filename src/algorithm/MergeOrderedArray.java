package algorithm;
/**
 * created by Sven
 * on 2019-12-02
 *
 * 合并两个有序数组
 */
public class MergeOrderedArray {
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

}
