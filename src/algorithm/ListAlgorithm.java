package algorithm;

import data.ListNode;

/**
 * created by Sven
 * on 2019-12-14
 *
 * 链表List的相关算法
 *
 */
public class ListAlgorithm {

    /**
     * created by Sven
     * on 2019
     *
     * 删除排序链表中的重复元素
     *
     * 给定一个排序链表，删除所有重复的元素，使得每个元素只出现一次。
     */
    public ListNode deleteDuplicates(ListNode head) {
        if (head == null || head.next == null)
            return head;
        ListNode moniterNode = head;
        ListNode indexNode = head.next;
        while (indexNode.next != null) {
            if (moniterNode.val != indexNode.val){
                moniterNode.next = indexNode;
                moniterNode = moniterNode.next;
            }
            indexNode = indexNode.next;
        }
        if(moniterNode.val != indexNode.val)
            moniterNode.next = indexNode;
        else
            moniterNode.next = null;
        return head;
    }

    /**
     * created by Sven
     * on 2019-12-01
     *
     * 合并两个有序链表
     */
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if(l1 == null)
            return l2;
        if(l2 == null)
            return l1;
        ListNode result = null,index = null;
        if(l1.val < l2.val){
            result = index = l1;
            l1 = l1.next;
        } else{
            result = index = l2;
            l2 = l2.next;
        }
        while (l1 != null && l2 != null){
            if(l1.val < l2.val){
                index.next = l1;
                l1 = l1.next;
            }else{
                index.next = l2;
                l2 = l2.next;
            }
            index = index.next;
        }
        if(l1 != null)
            index.next = l1;
        else
            index.next = l2;
        return result;
    }
}
