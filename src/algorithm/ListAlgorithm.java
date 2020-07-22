package algorithm;

import data.ListNode;

import java.util.*;

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

    /**
     * 相交链表
     *
     * 编写一个程序，找到两个单链表相交的起始节点。
     */
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if(headA == null || headB == null)
            return null;
        ListNode indexNode = headA;
        Set<ListNode> aNodeSet = new HashSet<>();
        while (indexNode != null){
            aNodeSet.add(indexNode);
            indexNode = indexNode.next;
        }
        indexNode = headB;
        while (indexNode != null){
            if(aNodeSet.contains(indexNode))
                return indexNode;
            indexNode = indexNode.next;
        }
        return null;
    }

    /**
     * 双指针法
     * @param headA
     * @param headB
     * @return
     */
    public ListNode getIntersectionNode2(ListNode headA, ListNode headB) {
        if(headA == null || headB == null)
            return null;
        ListNode indexNodeA = headA,indexNodeB = headB;
        boolean indexNodeAInA = true,indexNodeBInB = true;
        while (indexNodeA != null && indexNodeB != null){
            if(indexNodeB == indexNodeA)
                return indexNodeA;
            if(indexNodeA.next == null){
                if(indexNodeAInA) {
                    indexNodeA = headB;
                    indexNodeAInA = false;
                }
                else{
                    indexNodeA = indexNodeA.next;
                }
            }else{
                indexNodeA = indexNodeA.next;
            }
            if(indexNodeB.next == null){
                if(indexNodeBInB){
                    indexNodeB = headA;
                    indexNodeBInB = false;
                }else{
                    indexNodeB = indexNodeB.next;
                }
            }else{
                indexNodeB = indexNodeB.next;
            }
        }
        return null;
    }

    /**
     * 移除链表元素
     *
     * 删除链表中等于给定值 val 的所有节点。
     *
     * @param head
     * @param val
     * @return
     */
    public ListNode removeElements(ListNode head, int val) {
        if(head == null)
            return null;
        ListNode indexNode = head;
        while (indexNode != null){
            if(indexNode.next.val == val){
                indexNode.next = indexNode.next.next;
            }else{
                indexNode = indexNode.next;
            }
        }
        if(head.val == val)
            head = head.next;
        return head;
    }


    /**
     * 876. 链表的中间结点
     * 给定一个带有头结点 head 的非空单链表，返回链表的中间结点。
     *
     * 如果有两个中间结点，则返回第二个中间结点。
     * @param head
     * @return
     */
    public ListNode middleNode(ListNode head) {
        if(head == null || head.next == null)
            return head;
        ListNode index = head;
        ListNode doubleStepIndex = head;
        while (true){
            if(doubleStepIndex.next == null)
                return index;
            if(doubleStepIndex.next.next == null)
                return index.next;
            index = index.next;
            doubleStepIndex = doubleStepIndex.next.next;
        }
    }

    /**
     * 445. 两数相加 II
     * 给你两个 非空 链表来代表两个非负整数。数字最高位位于链表开始位置。它们的每个节点只存储一位数字。将这两数相加会返回一个新的链表。
     *
     * 你可以假设除了数字 0 之外，这两个数字都不会以零开头
     *
     * @param l1
     * @param l2
     * @return
     */
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        Stack<Integer> stack1 = new Stack<>();
        Stack<Integer> stack2 = new Stack<>();
        while (l1 != null){
            stack1.push(l1.val);
            l1 = l1.next;
        }
        while (l2 != null){
            stack2.push(l2.val);
            l2 = l2.next;
        }
        int carry = 0;
        ListNode head = null;
        while (!stack1.isEmpty() || !stack2.isEmpty() || carry > 0){
            int sum = carry;
            sum += stack1.isEmpty()?0:stack1.pop();
            sum += stack2.isEmpty() ? 0:stack2.pop();
            ListNode node = new ListNode(sum % 10);
            node.next = head;
            head = node;
            carry = sum /10;
        }
        return head;
    }

    /**
     * 23. 合并K个排序链表
     *
     * 合并 k 个排序链表，返回合并后的排序链表。请分析和描述算法的复杂度。
     *
     * @param lists
     * @return
     */
    public ListNode mergeKLists(ListNode[] lists) {
        if(lists == null || lists.length == 0)
            return null;
        PriorityQueue<ListNode> queue = new PriorityQueue<>(new Comparator<ListNode>() {
            @Override
            public int compare(ListNode o1, ListNode o2) {
                return (o1.val - o2.val);
            }
        });
        ListNode dummy = new ListNode(-1);
        ListNode current = dummy;
        for(int i = 0; i < lists.length;i++){
            ListNode head = lists[i];
            if(head!= null)
                queue.add(head);
        }
        while (queue.size() > 0){
            ListNode node = queue.poll();
            current.next = node;
            current = current.next;
            if(node.next != null){
                queue.add(node.next);
            }
        }
        current.next = null;
        return dummy.next;


    }

    /**
     * 面试题 02.03. 删除中间节点
     *
     * 实现一种算法，删除单向链表中间的某个节点（除了第一个和最后一个节点，不一定是中间节点），假定你只能访问该节点。
     *
     * @param node
     */
    public void deleteNode(ListNode node) {
        if(node  == null)
            return;
        if(node.next == null){
            node = null;
            return;
        }
        node.val = node.next.val;
        node.next = node.next.next;
    }

    /**
     * 面试题22. 链表中倒数第k个节点
     * 输入一个链表，输出该链表中倒数第k个节点。为了符合大多数人的习惯，本题从1开始计数，即链表的尾节点是倒数第1个节点。例如，一个链表有6个节点，从头节点开始，它们的值依次是1、2、3、4、5、6。这个链表的倒数第3个节点是值为4的节点。
     *
     * @param head
     * @param k
     * @return
     */
    public ListNode getKthFromEnd(ListNode head, int k) {
        if(head == null)
            return null;
        Queue<ListNode> queue = new LinkedList<>();
        ListNode index = head;
        while (index != null){
            if(queue.size() == k)
                queue.poll();
            queue.offer(index);
            index = index.next;
        }
        if(queue.size() == k)
            return queue.poll();
        else
            return null;
    }

    public ListNode getKthFromEnd_V2(ListNode head, int k) {
        if(head == null)
            return null;
        ListNode index = head;
        ListNode result = head;
        int i = 0;
        while (index != null){
            if(i < k){
                i++;
            }else{
                result = result.next;
            }
            index = index.next;
        }
        return result;
    }

    /**
     * 25. K 个一组翻转链表
     * 给你一个链表，每 k 个节点一组进行翻转，请你返回翻转后的链表。
     *
     * k 是一个正整数，它的值小于或等于链表的长度。
     *
     * 如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。
     *
     * @param head
     * @param k
     * @return
     */
    public ListNode reverseKGroup(ListNode head, int k) {
        ListNode hair = new ListNode(0);
        hair.next = head;
        ListNode pre = hair;
        ListNode end = hair;
        while(end.next != null){
            for(int i = 0; i<k && end != null;i++) end = end.next;
            if(end == null)break;
            ListNode start = pre.next;
            ListNode next = end.next;
            end.next = null;
            pre.next = reverseK(start);
            start.next = next;
            pre = start;
            end = pre;
        }
        return hair.next;
    }

    private ListNode reverseK(ListNode head){
        ListNode pre = null;
        ListNode curr = head;
        while (curr != null){
            ListNode next = curr.next;
            curr.next = pre;
            pre = curr;
            curr = next;
        }
        return pre;
    }

    /**
     * 面试题 02.01. 移除重复节点
     *
     * 编写代码，移除未排序链表中的重复节点。保留最开始出现的节点。
     *
     * @param head
     * @return
     */
    public ListNode removeDuplicateNodes(ListNode head) {
        if(head == null)
            return null;
        HashSet<Integer> dict = new HashSet<>();
        ListNode indexNode = head;
        dict.add(head.val);
        while (indexNode.next != null){
            if(dict.contains(indexNode.next.val)){
                indexNode.next = indexNode.next.next;
            }else{
                dict.add(indexNode.next.val);
                indexNode = indexNode.next;
            }
        }
        return head;
    }

    /**
     * 120. 三角形最小路径和
     *
     * 给定一个三角形，找出自顶向下的最小路径和。每一步只能移动到下一行中相邻的结点上。
     *
     * 相邻的结点 在这里指的是 下标 与 上一层结点下标 相同或者等于 上一层结点下标 + 1 的两个结点。
     *
     *  
     * @param triangle
     * @return
     */
    public int minimumTotal(List<List<Integer>> triangle) {
        if(triangle == null || triangle.size() == 0)
            return 0;
        int n = triangle.size();
        int [][] f= new int[n][n];
        f[0][0] = triangle.get(0).get(0);
        for(int i = 1; i < n; i++){
            f[i][0] = f[i - 1][0] + triangle.get(i).get(0);
            for(int j = 1; j < i ; j++){
                f[i][j] = Math.min(f[i-1][j-1],f[i-1][j]) + triangle.get(i).get(j);
            }
            f[i][i] = f[i-1][i-1] + triangle.get(i).get(i);
        }
        int minTotal = f[n-1][0];
        for(int i = 1; i < n; i++){
            minTotal = Math.min(minTotal,f[n-1][i]);
        }
        return minTotal;
    }

}
