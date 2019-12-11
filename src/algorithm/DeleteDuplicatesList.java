package algorithm;

import data.ListNode;

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
