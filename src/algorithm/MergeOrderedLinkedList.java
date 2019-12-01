package algorithm;

import data.ListNode;

/**
 * created by Sven
 * on 2019-12-01
 *
 * 合并两个有序链表
 */
public class MergeOrderedLinkedList {
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
