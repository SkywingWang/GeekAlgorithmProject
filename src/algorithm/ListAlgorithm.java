package algorithm;

import data.ListNode;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * created by Sven
 * on 2019-12-14
 * <p>
 * 链表List的相关算法
 */
public class ListAlgorithm {

    /**
     * created by Sven
     * on 2019
     * <p>
     * 删除排序链表中的重复元素
     * <p>
     * 给定一个排序链表，删除所有重复的元素，使得每个元素只出现一次。
     */
    public ListNode deleteDuplicates(ListNode head) {
        if (head == null || head.next == null)
            return head;
        ListNode moniterNode = head;
        ListNode indexNode = head.next;
        while (indexNode.next != null) {
            if (moniterNode.val != indexNode.val) {
                moniterNode.next = indexNode;
                moniterNode = moniterNode.next;
            }
            indexNode = indexNode.next;
        }
        if (moniterNode.val != indexNode.val)
            moniterNode.next = indexNode;
        else
            moniterNode.next = null;
        return head;
    }

    /**
     * created by Sven
     * on 2019-12-01
     * <p>
     * 合并两个有序链表
     */
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1 == null)
            return l2;
        if (l2 == null)
            return l1;
        ListNode result = null, index = null;
        if (l1.val < l2.val) {
            result = index = l1;
            l1 = l1.next;
        } else {
            result = index = l2;
            l2 = l2.next;
        }
        while (l1 != null && l2 != null) {
            if (l1.val < l2.val) {
                index.next = l1;
                l1 = l1.next;
            } else {
                index.next = l2;
                l2 = l2.next;
            }
            index = index.next;
        }
        if (l1 != null)
            index.next = l1;
        else
            index.next = l2;
        return result;
    }

    /**
     * created by Sven
     * on 2019-12-14
     * <p>
     * 杨辉三角
     * <p>
     * 给定一个非负整数 numRows，生成杨辉三角的前 numRows 行。
     */
    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> result = new ArrayList<>();
        if (numRows <= 0)
            return result;
        for (int i = 0; i < numRows; i++) {
            Integer[] tmp = new Integer[i + 1];
            if (i == 0) {
                tmp[0] = 1;
            } else if (i == 1) {
                tmp[0] = 1;
                tmp[1] = 1;
            } else {
                tmp[0] = 1;
                tmp[i] = 1;
                for (int j = 1; j < i; j++) {
                    tmp[j] = result.get(i - 1).get(j - 1) + result.get(i - 1).get(j);
                }
            }
            result.add(Arrays.asList(tmp));
        }
        return result;
    }

    /**
     * created by Sven
     * on 2019-12-23
     *
     * 杨辉三角
     *
     * 给定一个非负索引 k，其中 k ≤ 33，返回杨辉三角的第 k 行
     */
    public List<Integer> getRow(int rowIndex) {
        List<Integer> result = new ArrayList<>();
        int pre = 1;
        if(rowIndex == 0)
            result.add(1);
        if(rowIndex == 1){
            result.add(1);
            result.add(1);
        }else{
            result.add(1);
            for(int i = 1; i <= rowIndex;i++){
                for(int j = 1; j < i;j++){
                    int tmp = result.get(j);
                    result.set(j,pre + result.get(j));
                    pre = tmp;
                }
                result.add(1);
            }
        }
        return result;
    }

    /**
     * 环形链表
     *
     * 给定一个链表，判断链表中是否有环。
     *
     * @param head
     * @return
     */
    public boolean hasCycle(ListNode head) {
        if(head == null || head.next == null)
            return false;
        ListNode slow = head;
        ListNode fast = head.next;
        while (slow != fast){
            if(fast.next == null || fast.next.next == null)
                return false;
            slow = slow.next;
            fast = fast.next.next;
        }
        return true;
    }
}
