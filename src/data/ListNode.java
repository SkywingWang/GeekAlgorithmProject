package data;

/**
 * created by Sven
 * on 2019-12-03
 * <p>
 * List数据结构
 */
public class ListNode {
    public int val;
    public ListNode next;

    public ListNode(int x) {
        val = x;
    }

    public ListNode(int val, ListNode next) {
        this.val = val;
        this.next = next;
    }
}
