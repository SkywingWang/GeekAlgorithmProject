package algorithm;

import data.ListNode;

/**
 * created by Sven
 * on 2019
 *
 * 删除排序链表中的重复元素
 *
 * 给定一个排序链表，删除所有重复的元素，使得每个元素只出现一次。
 */
public class DeleteDuplicatesList {
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
}
