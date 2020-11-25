package algorithm;


import data.ListNode;
import data.Node;
import data.TreeNode;
import data.UnionFind;
import javafx.util.Pair;

import java.util.*;

/**
 * created by Sven
 * on 2019-12-14
 * <p>
 * 二叉树的相关算法
 */
public class BinaryTreeAlgorithm {

    /**
     * created by Sven
     * on 2019-12-12
     * <p>
     * 相同的树
     * <p>
     * 给定两个二叉树，编写一个函数来检验它们是否相同。
     * <p>
     * 如果两个树在结构上相同，并且节点具有相同的值，则认为它们是相同的。
     */
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null && q == null)
            return true;
        if (p == null || q == null)
            return false;
        if (p.val != q.val)
            return false;
        return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
    }


    /**
     * created by Sven
     * on 2019-12-12
     * <p>
     * 镜像树 对称二叉树
     * <p>
     * 给定一个二叉树，检查它是否是镜像对称的。
     * <p>
     * 例如，二叉树 [1,2,2,3,4,4,3] 是对称的。
     */
    public boolean isSymmetric(TreeNode root) {
        return isMirror(root, root);
    }

    public boolean isMirror(TreeNode t1, TreeNode t2) {
        if (t1 == null && t2 == null) return true;
        if (t1 == null || t2 == null) return false;
        return (t1.val == t2.val)
                && isMirror(t1.right, t2.left)
                && isMirror(t1.left, t2.right);
    }

    /**
     * created by Sven
     * on 2019-12-14
     * <p>
     * 二叉树的最大深度
     * <p>
     * 给定一个二叉树，找出其最大深度。
     * <p>
     * 二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。
     * <p>
     * 说明: 叶子节点是指没有子节点的节点。
     */
    public int maxDepth(TreeNode root) {
        if (root == null)
            return 0;
        if (root.left == null && root.right == null)
            return 1;
        return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
    }

    /**
     * created by Sven
     * on 2019-12-16
     * <p>
     * 二叉树的层次遍历
     * <p>
     * 给定一个二叉树，返回其节点值自底向上的层次遍历。 （即按从叶子节点所在层到根节点所在的层，逐层从左向右遍历）
     */
    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        List<List<Integer>> ans = new ArrayList<>();
        DFS(root, 0, ans);
        return ans;
    }

    private void DFS(TreeNode root, int level, List<List<Integer>> ans) {
        if (root == null) {
            return;
        }

        //当前层如果还没有元素，先new一个空列表
        if (ans.size() <= level) {
            ans.add(0, new ArrayList<>());
        }
        ans.get(ans.size() - 1 - level).add(root.val);
        DFS(root.left, level + 1, ans);
        DFS(root.right, level + 1, ans);
    }

    /**
     * 将有序数组转换为二叉搜索树
     * <p>
     * 将一个按照升序排列的有序数组，转换为一棵高度平衡二叉搜索树。
     * <p>
     * 本题中，一个高度平衡二叉树是指一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过 1。
     * <p>
     * 二叉搜索树 实质相当于 二分法
     * 每次都将nums[0 ... length-1]从中间截断，分成两半。
     * 取中间的nums[mid]为root.val :mid = (l+r+1)/2
     * root.letf 根据左半边数据按上述思想生成
     * root.right 根据右半边数据按上述思想生成
     */

    public TreeNode sortedArrayToBST(int[] nums) {
        return createTree(nums, 0, nums.length - 1);
    }

    private TreeNode createTree(int[] tmp, int l, int r) {
        if (l == r)
            return new TreeNode(tmp[l]);
        else if (l > r)
            return null;
        int mid = (l + r + 1) >> 1;
        TreeNode t = new TreeNode(tmp[mid]);
        t.left = createTree(tmp, l, mid - 1);
        t.right = createTree(tmp, mid + 1, r);
        return t;
    }


    /**
     * created by Sven
     * on 2019-12-16
     * <p>
     * 平衡二叉树
     * <p>
     * 给定一个二叉树，判断它是否是高度平衡的二叉树。
     * <p>
     * 本题中，一棵高度平衡二叉树定义为：
     * <p>
     * 一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过1。
     */
    public boolean isBalanced(TreeNode root) {
        return depth(root) != -1;
    }

    private int depth(TreeNode root) {
        if (root == null) return 0;
        int left = depth(root.left);
        if (left == -1) return -1;
        int right = depth(root.right);
        if (right == -1) return -1;
        return Math.abs(left - right) < 2 ? Math.max(left, right) + 1 : -1;
    }

    /**
     * created by Sven
     * on 2019-12-20
     * <p>
     * 给定一个二叉树，找出其最小深度。
     * <p>
     * 最小深度是从根节点到最近叶子节点的最短路径上的节点数量。
     * <p>
     * 说明: 叶子节点是指没有子节点的节点
     *
     * 递归
     */

    public int minDepth(TreeNode root) {
        if (root == null)
            return 0;
        if (root.left == null && root.right == null)
            return 1;
        int min_depth = Integer.MAX_VALUE;
        if(root.left != null)
            min_depth = Math.min(minDepth(root.left),min_depth);
        if(root.right != null)
            min_depth = Math.min(minDepth(root.right),min_depth);
        return min_depth + 1;
    }

    /**
     * created by Sven
     * on 2019-12-20
     * <p>
     * 给定一个二叉树，找出其最小深度。
     * <p>
     * 最小深度是从根节点到最近叶子节点的最短路径上的节点数量。
     * <p>
     * 说明: 叶子节点是指没有子节点的节点
     *
     * 深度优先搜索
     */
    public int minDepth_deep(TreeNode root) {
        LinkedList<Pair<TreeNode,Integer>> stack = new LinkedList<>();
        if(root == null)
            return 0;
        else
            stack.add(new Pair(root,1));
        int min_depth = Integer.MAX_VALUE;
        while (!stack.isEmpty()){
            Pair<TreeNode,Integer> current = stack.pollLast();
            root = current.getKey();
            int current_depth = current.getValue();
            if((root.left == null) && (root.right == null))
                min_depth = Math.min(min_depth,current_depth);
            if(root.left != null)
                stack.add(new Pair<>(root.left,current_depth + 1));
            if(root.right != null)
                stack.add(new Pair<>(root.right,current_depth + 1));
        }
        return min_depth;

    }

    /**
     * created by Sven
     * on 2019-12-20
     *
     * 给定一个二叉树和一个目标和，判断该树中是否存在根节点到叶子节点的路径，这条路径上所有节点值相加等于目标和。
     *
     * 说明: 叶子节点是指没有子节点的节点。
     */
    public boolean hasPathSum(TreeNode root, int sum) {
        if(root == null)
            return false;
        if(root.left == null && root.right == null){
            if(root.val != sum)
                return false;
            else
                return true;
        }
        return hasPathSum(root.left,sum - root.val) || hasPathSum(root.right,sum-root.val);

    }

    /**
     * 199. 二叉树的右视图
     * 给定一棵二叉树，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。
     * @param root
     * @return
     */
    public List<Integer> rightSideView(TreeNode root) {
        Map<Integer,Integer> rightMostValueAtDepth = new HashMap<>();
        int max_depth = -1;
        Queue<TreeNode> nodeQueue = new LinkedList<>();
        Queue<Integer> depthQueue = new LinkedList<>();
        nodeQueue.add(root);
        depthQueue.add(0);
        while (!nodeQueue.isEmpty()){
            TreeNode node = nodeQueue.remove();
            int depth = depthQueue.remove();

            if(node != null){
                max_depth = Math.max(max_depth,depth);
                rightMostValueAtDepth.put(depth,node.val);

                nodeQueue.add(node.left);
                nodeQueue.add(node.right);
                depthQueue.add(depth + 1);
                depthQueue.add(depth + 1);
            }
        }
        List<Integer> rightView = new ArrayList<>();
        for(int depth = 0; depth <= max_depth; depth++){
            rightView.add(rightMostValueAtDepth.get(depth));
        }
        return rightView;
    }

    /**
     * 98. 验证二叉搜索树
     * 给定一个二叉树，判断其是否是一个有效的二叉搜索树。
     *
     * 假设一个二叉搜索树具有如下特征：
     *
     * 节点的左子树只包含小于当前节点的数。
     * 节点的右子树只包含大于当前节点的数。
     * 所有左子树和右子树自身必须也是二叉搜索树。
     *
     * @param root
     * @return
     */
    public boolean isValidBST(TreeNode root) {
        return isValidBSTHelper(root,null,null);
    }

    private boolean isValidBSTHelper(TreeNode node,Integer lower,Integer upper){
        if(node == null)
            return true;
        if(lower != null && node.val <= lower) return false;
        if(upper != null && node.val >= upper) return false;
        if(!isValidBSTHelper(node.left,lower,node.val)) return false;
        if(!isValidBSTHelper(node.right,node.val,upper)) return false;
        return true;

    }

    /**
     * 572. 另一个树的子树
     * 给定两个非空二叉树 s 和 t，检验 s 中是否包含和 t 具有相同结构和节点值的子树。s 的一个子树包括 s 的一个节点和这个节点的所有子孙。s 也可以看做它自身的一棵子树。
     *
     * @param s
     * @param t
     * @return
     */
    public boolean isSubtree(TreeNode s, TreeNode t) {
        if(t==null) return true;
        if(s==null) return false;
        return isSubtree(s.left,t) || isSubtree(s.right,t) || isSameTreeSub(s,t);
    }

    private boolean isSameTreeSub(TreeNode s,TreeNode t){
        if(s == null && t == null) return true;
        if(s == null || t == null) return false;
        if(s.val != t.val) return false;
        return isSameTreeSub(s.left,t.left) && isSameTreeSub(s.right,t.right);
    }

    /**
     * 236. 二叉树的最近公共祖先
     * 给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。
     *
     * 百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”
     *
     * @param root
     * @param p
     * @param q
     * @return
     */
    private TreeNode resultTreeNode;
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        lowestCommonAncestorDFS(root,p,q);
        return resultTreeNode;
    }
    private boolean lowestCommonAncestorDFS(TreeNode root,TreeNode p,TreeNode q){
        if(root == null) return false;
        boolean lson = lowestCommonAncestorDFS(root.left,p,q);
        boolean rson = lowestCommonAncestorDFS(root.right,p,q);
        if((lson && rson) || ((root.val == p.val || root.val == q.val) && (lson || rson))){
            resultTreeNode = root;
        }
        return lson || rson||(root.val == p.val || root.val == q.val);
    }

    /**
     * 102. 二叉树的层序遍历
     *
     * 给你一个二叉树，请你返回其按 层序遍历 得到的节点值。 （即逐层地，从左到右访问所有节点）。
     * @param root
     * @return
     */
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if(root == null)
            return result;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()){
            int size = queue.size();
            List<Integer> list = new ArrayList<>();
            while (size > 0){
                TreeNode curr = queue.poll();
                list.add(curr.val);
                size--;
                if(curr.left!= null)
                    queue.offer(curr.left);
                if(curr.right != null){
                    queue.offer(curr.right);
                }
            }
            result.add(list);
        }
        return result;
    }

    /**
     * 105. 从前序与中序遍历序列构造二叉树
     *
     * 根据一棵树的前序遍历与中序遍历构造二叉树。
     *
     * 注意:
     * 你可以假设树中没有重复的元素。
     *
     * @param preorder
     * @param inorder
     * @return
     */
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        if(preorder == null || preorder.length == 0)
            return null;
        TreeNode root = new TreeNode(preorder[0]);
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        int inorderIndex = 0;
        for(int i = 1; i < preorder.length; i++){
            int preOrderVal = preorder[i];
            TreeNode node = stack.peek();
            if(node.val != inorder[inorderIndex]){
                node.left = new TreeNode(preOrderVal);
                stack.push(node.left);
            }else {
                while (!stack.isEmpty() && stack.peek().val == inorder[inorderIndex]){
                    node = stack.pop();
                    inorderIndex++;
                }
                node.right = new TreeNode(preOrderVal);
                stack.push(node.right);
            }
        }
        return root;
    }

    public boolean isSymmetricV2(TreeNode root) {
       return checkIsSymmetric(root,root);
    }

    private boolean checkIsSymmetric(TreeNode leftNode,TreeNode rightNode){
        if(leftNode == null && rightNode == null)
            return true;
        if(leftNode == null || rightNode == null)
            return false;
        return leftNode.val == rightNode.val && checkIsSymmetric(leftNode.left,rightNode.right) && checkIsSymmetric(leftNode.right,rightNode.left);
    }

    /**
     * 1028. 从先序遍历还原二叉树
     *
     * 我们从二叉树的根节点 root 开始进行深度优先搜索。
     *
     * 在遍历中的每个节点处，我们输出 D 条短划线（其中 D 是该节点的深度），然后输出该节点的值。（如果节点的深度为 D，则其直接子节点的深度为 D + 1。根节点的深度为 0）。
     *
     * 如果节点只有一个子节点，那么保证该子节点为左子节点。
     *
     * 给出遍历输出 S，还原树并返回其根节点 root。
     *
     * @param S
     * @return
     */
    public TreeNode recoverFromPreorder(String S) {
        Deque<TreeNode> path = new LinkedList<TreeNode>();
        int pos = 0; //字符在字符串中的位置的索引
        while (pos < S.length()){
            int level = 0;
            while (S.charAt(pos) == '-'){
                ++level;
                ++pos;
            }
            int value = 0;
            while (pos < S.length() && Character.isDigit(S.charAt(pos))){
                value = value * 10 + (S.charAt(pos) - '0');
                ++pos;
            }
            TreeNode node = new TreeNode(value);
            if(level == path.size()){
                if(!path.isEmpty()){
                    path.peek().left = node;
                }
            }else {
                while (level != path.size()){
                    path.pop();
                }
                path.peek().right = node;
            }
            path.push(node);
        }
        while (path.size() > 1){
            path.pop();
        }
        return path.peek();
    }

    /**
     * 124. 二叉树中的最大路径和
     *
     * 给定一个非空二叉树，返回其最大路径和。
     *
     * 本题中，路径被定义为一条从树中任意节点出发，达到任意节点的序列。该路径至少包含一个节点，且不一定经过根节点。
     *
     * @param root
     * @return
     */
    private int maxSum = Integer.MIN_VALUE;
    public int maxPathSum(TreeNode root) {
        maxGain(root);
        return maxSum;
    }
    public int maxGain(TreeNode node){
        if(node == null)
            return 0;
        int leftGain = Math.max(maxGain(node.left),0);
        int rightGain = Math.max(maxGain(node.right),0);
        int priceNewPath = node.val + leftGain + rightGain;
        maxSum = Math.max(maxSum,priceNewPath);
        return node.val + Math.max(leftGain,rightGain);
    }

    /**
     * 41. 缺失的第一个正数
     *
     * 给你一个未排序的整数数组，请你找出其中没有出现的最小的正整数。
     *
     * @param nums
     * @return
     */
    public int firstMissingPositive(int[] nums) {
        if(nums == null || nums.length == 0)
            return 1;
        int n = nums.length;
        for(int i = 0; i < n; i++){
            if(nums[i] < 0){
                nums[i] = n + 1;
            }
        }
        for(int i = 0; i < n; i++){
            int num = Math.abs(nums[i]);
            if (num <= n) {
                nums[num - 1] = -Math.abs(nums[num - 1]);
            }
        }
        for(int i = 0;i < n; i++){
            if(nums[i] > 0){
                return i + 1;
            }
        }
        return n+1;
    }

    /**
     * 96. 不同的二叉搜索树
     *
     * 给定一个整数 n，求以 1 ... n 为节点组成的二叉搜索树有多少种？
     *
     * @param n
     * @return
     */
    public int numTrees(int n) {
        if(n <= 0)
            return 0;
        int[] G = new int[n + 1];
        G[0] = 1;
        G[1] = 1;
        for(int i = 2; i <= n; i++){
            for(int j = 1; j <= i; j++){
                G[i] += G[j - 1] * G[i - j];
            }
        }
        return G[n];
    }

    /**
     * 785. 判断二分图
     *
     * @param graph
     * @return
     */
    private static final int UNCOLORED = 0;
    private static final int RED = 1;
    private static final int GREEN = 2;
    private int[] color;
    private boolean valid;

    public boolean isBipartite(int[][] graph) {
        int n = graph.length;
        valid = true;
        color = new int[n];
        Arrays.fill(color,UNCOLORED);
        for(int i = 0; i < n && valid; i++){
            if(color[i]==UNCOLORED){
                isBipartiteDFS(i,RED,graph);
            }
        }
        return valid;
    }

    private void isBipartiteDFS(int node,int c,int[][] graph){
        color[node] = c;
        int cNei = c == RED ? GREEN : RED;
        for(int neighbor : graph[node]){
            if(color[neighbor] == UNCOLORED){
                isBipartiteDFS(neighbor,cNei,graph);
                if(!valid)
                    return;
            }else if(color[neighbor] != cNei){
                valid = false;
                return;
            }
        }
    }

    public void flatten(TreeNode root) {
        List<TreeNode> list = new ArrayList<TreeNode>();
        preorderTraversal(root, list);
        int size = list.size();
        for (int i = 1; i < size; i++) {
            TreeNode prev = list.get(i - 1), curr = list.get(i);
            prev.left = null;
            prev.right = curr;
        }
    }

    public void preorderTraversal(TreeNode root, List<TreeNode> list) {
        if (root != null) {
            list.add(root);
            preorderTraversal(root.left, list);
            preorderTraversal(root.right, list);
        }
    }

    /**
     * 337. 打家劫舍 III
     *
     * 在上次打劫完一条街道之后和一圈房屋后，小偷又发现了一个新的可行窃的地区。这个地区只有一个入口，我们称之为“根”。 除了“根”之外，每栋房子有且只有一个“父“房子与之相连。一番侦察之后，聪明的小偷意识到“这个地方的所有房屋的排列类似于一棵二叉树”。 如果两个直接相连的房子在同一天晚上被打劫，房屋将自动报警。
     *
     * 计算在不触动警报的情况下，小偷一晚能够盗取的最高金额。
     *
     * @param root
     * @return
     */
    Map<TreeNode,Integer> f = new HashMap<>();
    Map<TreeNode,Integer> g = new HashMap<>();

    public int rob(TreeNode root) {
        dfsRob(root);
        return Math.max(f.getOrDefault(root,0),g.getOrDefault(root,0));
    }
    public void dfsRob(TreeNode node){
        if(node == null)
            return;
        dfsRob(node.left);
        dfsRob(node.right);
        f.put(node,node.val + g.getOrDefault(node.left,0) + g.getOrDefault(node.right,0));
        g.put(node,Math.max(f.getOrDefault(node.left,0),g.getOrDefault(node.left,0)) + Math.max(f.getOrDefault(node.right,0),g.getOrDefault(node.right,0)));
    }

    /**
     * 99. 恢复二叉搜索树
     *
     * 二叉搜索树中的两个节点被错误地交换。
     * 请在不改变其结构的情况下，恢复这棵树。
     *
     * @param root
     */
    public void recoverTree(TreeNode root) {
        List<Integer> nums = new ArrayList<>();
        inorderRecoverTree(root,nums);
        int[] swapped = findTwoSwapperdRecoverTree(nums);
        recoverRecoverTree(root,2,swapped[0],swapped[1]);
    }

    private void inorderRecoverTree(TreeNode root,List<Integer> nums){
        if(root == null)
            return;
        inorderRecoverTree(root.left,nums);
        nums.add(root.val);
        inorderRecoverTree(root.right,nums);
    }

    private int[] findTwoSwapperdRecoverTree(List<Integer> nums){
        int n = nums.size();
        int x = -1, y = -1;
        for(int i = 0; i < n - 1; i++){
            if(nums.get(i + 1) < nums.get(i)){
                y = nums.get(i + 1);
                if( x == -1){
                    x = nums.get(i);
                }else
                    break;
            }
        }
        return new int[]{x,y};
    }
    private void recoverRecoverTree(TreeNode root,int count,int x,int y){
        if(root != null){
            if(root.val == x || root.val == y){
                root.val = root.val == x ? y : x;
                if(--count == 0)
                    return;
            }
            recoverRecoverTree(root.right,count,x,y);
            recoverRecoverTree(root.left,count,x,y);
        }
    }

    /**
     * 109. 有序链表转换二叉搜索树
     * 给定一个单链表，其中的元素按升序排序，将其转换为高度平衡的二叉搜索树。
     *
     * 本题中，一个高度平衡二叉树是指一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过 1。
     *
     *
     * @param head
     * @return
     */
    public TreeNode sortedListToBST(ListNode head) {
        return buildTreeSortedListToBST(head,null);
    }

    private TreeNode buildTreeSortedListToBST(ListNode left,ListNode right){
        if(left == right)
            return null;
        ListNode mid = getMedianSortedListToBST(left,right);
        TreeNode root = new TreeNode(mid.val);
        root.left = buildTreeSortedListToBST(left,mid);
        root.right = buildTreeSortedListToBST(mid.next,right);
        return root;
    }

    private ListNode getMedianSortedListToBST(ListNode left,ListNode right){
        ListNode fast = left;
        ListNode slow = left;
        while (fast != right && fast.next != right){
            fast = fast.next;
            fast = fast.next;
            slow = slow.next;
        }
        return slow;
    }

    /**
     * 94. 二叉树的中序遍历
     * 给定一个二叉树，返回它的中序 遍历。
     * @param root
     * @return
     */
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        inorderMiddle(root,res);
        return res;
    }

    private void inorderMiddle(TreeNode root,List<Integer> res){
        if(root == null){
            return;
        }
        inorderMiddle(root.left, res);
        res.add(root.val);
        inorderMiddle(root.right,res);
    }

    /**
     * 226. 翻转二叉树
     * 翻转一棵二叉树。
     * @param root
     * @return
     */
    public TreeNode invertTree(TreeNode root) {
        if(root ==null)
            return null;
        TreeNode left = invertTree(root.left);
        TreeNode right = invertTree(root.right);
        root.left = right;
        root.right = left;
        return root;
    }

    /**
     * 685. 冗余连接 II
     *
     * 在本问题中，有根树指满足以下条件的有向图。该树只有一个根节点，所有其他节点都是该根节点的后继。每一个节点只有一个父节点，除了根节点没有父节点。
     * 输入一个有向图，该图由一个有着N个节点 (节点值不重复1, 2, ..., N) 的树及一条附加的边构成。附加的边的两个顶点包含在1到N中间，这条附加的边不属于树中已存在的边。
     * 结果图是一个以边组成的二维数组。 每一个边 的元素是一对 [u, v]，用以表示有向图中连接顶点 u 和顶点 v 的边，其中 u 是 v 的一个父节点。
     * 返回一条能删除的边，使得剩下的图是有N个节点的有根树。若有多个答案，返回最后出现在给定二维数组的答案。
     * @param edges
     * @return
     */
    public int[] findRedundantDirectedConnection(int[][] edges) {
        int nodesCount = edges.length;
        UnionFind uf = new UnionFind(nodesCount + 1);
        int[] parent = new int[nodesCount + 1];
        for(int i = 1; i <= nodesCount; i++){
            parent[i] = i;
        }
        int conflict = -1;
        int cycle = -1;
        for(int i = 0; i < nodesCount; i++){
            int[] edge = edges[i];
            int node1 = edge[0], node2 = edge[1];
            if (parent[node2] != node2) {
                conflict = i;
            } else {
                parent[node2] = node1;
                if (uf.find(node1) == uf.find(node2)) {
                    cycle = i;
                } else {
                    uf.union(node1, node2);
                }
            }
        }
        if (conflict < 0) {
            int[] redundant = {edges[cycle][0], edges[cycle][1]};
            return redundant;
        } else {
            int[] conflictEdge = edges[conflict];
            if (cycle >= 0) {
                int[] redundant = {parent[conflictEdge[1]], conflictEdge[1]};
                return redundant;
            } else {
                int[] redundant = {conflictEdge[0], conflictEdge[1]};
                return redundant;
            }
        }
    }

    /**
     * 404. 左叶子之和
     * 计算给定二叉树的所有左叶子之和。
     * @param root
     * @return
     */
    public int sumOfLeftLeaves(TreeNode root) {
        if(root == null){
            return 0;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        int ans = 0;
        while (!queue.isEmpty()){
            TreeNode node = queue.poll();
            if(node.left != null){
                if(isLeafNodeSum(node.left)){
                    ans += node.left.val;
                }else{
                    queue.offer(node.left);
                }
            }
            if(node.right != null){
                if(!isLeafNodeSum(node.right)){
                    queue.offer(node.right);
                }
            }
        }
        return ans;
    }

    private boolean isLeafNodeSum(TreeNode node){
        return node.left == null && node.right == null;
    }

    /**
     * 538. 把二叉搜索树转换为累加树
     *
     * 给定一个二叉搜索树（Binary Search Tree），
     * 把它转换成为累加树（Greater Tree)，
     * 使得每个节点的值是原来的节点值加上所有大于它的节点值之和。
     *
     * @param root
     * @return
     */
    int sumConvertBST = 0;
    public TreeNode convertBST(TreeNode root) {
        if(root != null){
            convertBST(root.right);
            sumConvertBST += root.val;
            root.val = sumConvertBST;
            convertBST(root.left);
        }
        return root;
    }

    /**
     * 968. 监控二叉树
     *
     * 给定一个二叉树，我们在树的节点上安装摄像头。
     *
     * 节点上的每个摄影头都可以监视其父对象、自身及其直接子对象。
     *
     * 计算监控树的所有节点所需的最小摄像头数量。
     *
     * @param root
     * @return
     */
    public int minCameraCover(TreeNode root) {
        int[] array = dfsMinCameraCover(root);
        return array[1];
    }

    private int[] dfsMinCameraCover(TreeNode root){
        if(root == null){
            return new int[]{Integer.MAX_VALUE / 2,0,0};
        }
        int[] leftArray = dfsMinCameraCover(root.left);
        int[] rightArray = dfsMinCameraCover(root.right);
        int[] array = new int[3];
        array[0] = leftArray[2] + rightArray[2] + 1;
        array[1] = Math.min(array[0],Math.min(leftArray[0] + rightArray[1] ,rightArray[0] + leftArray[1]));
        array[2] = Math.min(array[0],leftArray[1] + rightArray[1]);
        return array;

    }

    /**
     * 617. 合并二叉树
     *
     * 给定两个二叉树，想象当你将它们中的一个覆盖到另一个上时，两个二叉树的一些节点便会重叠。
     *
     * 你需要将他们合并为一个新的二叉树。合并的规则是如果两个节点重叠，那么将他们的值相加作为节点合并后的新值，
     *
     * 否则不为 NULL 的节点将直接作为新二叉树的节点。
     *
     * @param t1
     * @param t2
     * @return
     */
    public TreeNode mergeTrees(TreeNode t1, TreeNode t2) {
        if(t1 == null){
            return t2;
        }
        if(t2 == null){
            return t1;
        }
        TreeNode merged = new TreeNode(t1.val + t2.val);
        merged.left = mergeTrees(t1.left,t2.left);
        merged.right = mergeTrees(t1.right,t2.right);
        return merged;
    }


    /**
     * 501. 二叉搜索树中的众数
     * 给定一个有相同值的二叉搜索树（BST），找出 BST 中的所有众数（出现频率最高的元素）。
     *
     * 假定 BST 有如下定义：
     *
     * 结点左子树中所含结点的值小于等于当前结点的值
     * 结点右子树中所含结点的值大于等于当前结点的值
     * 左子树和右子树都是二叉搜索树
     * 例如：
     * 给定 BST [1,null,2,2],
     *
     * @param root
     * @return
     */
    int baseFindMode,countFindMode,maxCountFindMode;
    List<Integer> answerFindMode = new ArrayList<>();
    public int[] findMode(TreeNode root) {
        TreeNode cur = root,pre = null;
        while(cur != null){
            if(cur.left == null){
                updateFindMode(cur.val);
                cur = cur.right;
                continue;
            }
            pre = cur.left;
            while (pre.right != null && pre.right != cur){
                pre = pre.right;
            }
            if(pre.right == null){
                pre.right = cur;
                cur = cur.left;
            }else{
                pre.right = null;
                updateFindMode(cur.val);
                cur = cur.right;
            }
        }
        int[] mode = new int[answerFindMode.size()];
        for(int i = 0; i < answerFindMode.size(); i++){
            mode[i] = answerFindMode.get(i);
        }
        return mode;
    }

    private void updateFindMode(int x){
        if(x == baseFindMode){
            countFindMode ++;
        }else{
            countFindMode = 1;
            baseFindMode = x;
        }
        if(countFindMode == maxCountFindMode){
            answerFindMode.add(baseFindMode);
        }
        if(countFindMode > maxCountFindMode){
            maxCountFindMode = countFindMode;
            answerFindMode.clear();
            answerFindMode.add(baseFindMode);
        }
    }

    /**
     * 106. 从中序与后序遍历序列构造二叉树
     *
     * 根据一棵树的中序遍历与后序遍历构造二叉树。
     *
     * 注意:
     * 你可以假设树中没有重复的元素。
     *
     * @param inorder
     * @param postorder
     * @return
     */
    public TreeNode buildTree_v2(int[] inorder, int[] postorder) {
        if(postorder == null || postorder.length == 0){
            return null;
        }
        TreeNode root = new TreeNode(postorder[postorder.length - 1]);
        Deque<TreeNode> stack = new LinkedList<>();
        stack.push(root);
        int inorderIndex = inorder.length - 1;
        for(int i = postorder.length - 2; i >= 0; i--){
            int postoderVal = postorder[i];
            TreeNode node = stack.peek();
            if(node.val != inorder[inorderIndex]){
                node.right = new TreeNode(postoderVal);
                stack.push(node.right);
            }else{
                while (!stack.isEmpty() && stack.peek().val == inorder[inorderIndex]) {
                    node = stack.pop();
                    inorderIndex--;
                }
                node.left = new TreeNode(postoderVal);
                stack.push(node.left);
            }
        }
        return root;
    }

    /**
     * 235. 二叉搜索树的最近公共祖先
     * 给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。
     *
     * 百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”
     *
     * 例如，给定如下二叉搜索树:  root = [6,2,8,0,4,7,9,null,null,3,5]
     *
     * @param root
     * @param p
     * @param q
     * @return
     */
    public TreeNode lowestCommonAncestor_V2(TreeNode root, TreeNode p, TreeNode q) {
        List<TreeNode> path_p = getPathLowestCommonAncestor(root,p);
        List<TreeNode> path_q = getPathLowestCommonAncestor(root,q);
        TreeNode ancestor = null;
        for(int i = 0; i < path_p.size() && i < path_q.size(); i++){
            if(path_p.get(i) == path_q.get(i)){
                ancestor = path_p.get(i);
            }else {
                break;
            }
        }
        return ancestor;
    }

    private List<TreeNode> getPathLowestCommonAncestor(TreeNode root,TreeNode target){
        List<TreeNode> path = new ArrayList<>();
        TreeNode node = root;
        while (node != target){
            path.add(node);
            if(target.val < node.val){
                node = node.left;
            }else {
                node = node.right;
            }
        }
        path.add(node);
        return path;
    }

    /**
     * 117. 填充每个节点的下一个右侧节点指针 II
     *
     * 填充它的每个 next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将 next 指针设置为 NULL。
     *
     * 初始状态下，所有 next 指针都被设置为 NULL。
     *
     * @param root
     * @return
     */
    public Node connect(Node root) {
        if(root == null){
            return null;
        }
        Queue<Node> queue = new LinkedList<Node>();
        queue.offer(root);
        while (!queue.isEmpty()){
            int n = queue.size();
            Node last = null;
            for(int i = 1; i <= n; i++){
                Node f = queue.poll();
                if(f.left != null){
                    queue.offer(f.left);
                }
                if(f.right != null){
                    queue.offer(f.right);
                }
                if(i != 1){
                    last.next = f;
                }
                last = f;
            }
        }
        return root;
    }

    /**
     * 145. 二叉树的后序遍历
     *
     * 给定一个二叉树，返回它的 后序 遍历。
     * @param root
     * @return
     */
    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if(root == null){
            return res;
        }
        Deque<TreeNode> stack = new LinkedList<>();
        TreeNode prev = null;
        while (root != null || !stack.isEmpty()){
            while (root != null){
                stack.push(root);
                root = root.left;
            }
            root = stack.pop();
            if(root.right == null || root.right == prev){
                res.add(root.val);
                prev = root;
                root = null;
            }else{
                stack.push(root);
                root = root.right;
            }
        }
        return res;
    }

    /**
     * 701. 二叉搜索树中的插入操作
     *
     * 给定二叉搜索树（BST）的根节点和要插入树中的值，将值插入二叉搜索树。 返回插入后二叉搜索树的根节点。 输入数据保证，新值和原始二叉搜索树中的任意节点值都不同。
     *
     * 注意，可能存在多种有效的插入方式，只要树在插入后仍保持为二叉搜索树即可。 你可以返回任意有效的结果。
     *
     *
     * @param root
     * @param val
     * @return
     */
    public TreeNode insertIntoBST(TreeNode root, int val) {
        if(root == null){
            return new TreeNode(val);
        }
        TreeNode pos = root;
        while (pos !=null){
            if(val < pos.val){
                if(pos.left == null){
                    pos.left = new TreeNode(val);
                    break;
                }else{
                    pos = pos.left;
                }
            }else{
                if(pos.right == null){
                    pos.right = new TreeNode(val);
                    break;
                }else{
                    pos = pos.right;
                }
            }
        }
        return root;
    }

    /**
     *
     * @param leaves
     * @return
     */
    public int minimumOperations(String leaves) {
        int n = leaves.length();
        int[][] f = new int[n][3];
        f[0][0] = leaves.charAt(0) == 'y'?1:0;
        f[0][1] = f[0][2] = f[1][2] = Integer.MAX_VALUE;
        for (int i = 1; i < n; ++i) {
            int isRed = leaves.charAt(i) == 'r' ? 1 : 0;
            int isYellow = leaves.charAt(i) == 'y' ? 1 : 0;
            f[i][0] = f[i - 1][0] + isYellow;
            f[i][1] = Math.min(f[i - 1][0], f[i - 1][1]) + isRed;
            if (i >= 2) {
                f[i][2] = Math.min(f[i - 1][1], f[i - 1][2]) + isYellow;
            }
        }
        return f[n - 1][2];
    }


    /**
     * 834. 树中距离之和
     *
     * 给定一个无向、连通的树。树中有 N 个标记为 0...N-1 的节点以及 N-1 条边 。
     *
     * 第 i 条边连接节点 edges[i][0] 和 edges[i][1] 。
     *
     * 返回一个表示节点 i 与其他所有节点距离之和的列表 ans。
     *
     * @param N
     * @param edges
     * @return
     */
    int[] ansSumOfDistancesInTree;
    int[] szSumOfDistancesInTree;
    int[] dpSumOfDistancesInTree;
    List<List<Integer>> graphSumOfDistancesInTree;
    public int[] sumOfDistancesInTree(int N, int[][] edges) {
        ansSumOfDistancesInTree = new int[N];
        szSumOfDistancesInTree = new int[N];
        dpSumOfDistancesInTree = new int[N];
        graphSumOfDistancesInTree = new ArrayList<List<Integer>>();
        for (int i = 0; i < N; ++i) {
            graphSumOfDistancesInTree.add(new ArrayList<Integer>());
        }
        for (int[] edge: edges) {
            int u = edge[0], v = edge[1];
            graphSumOfDistancesInTree.get(u).add(v);
            graphSumOfDistancesInTree.get(v).add(u);
        }
        dfsSumOfDistancesInTree(0, -1);
        dfsSumOfDistancesInTree_2(0, -1);
        return ansSumOfDistancesInTree;
    }

    private void dfsSumOfDistancesInTree(int u,int f){
        szSumOfDistancesInTree[u] = 1;
        dpSumOfDistancesInTree[u] = 0;
        for(int v : graphSumOfDistancesInTree.get(u)){
            if(v == f){
                continue;
            }
            dfsSumOfDistancesInTree(v,u);
            dpSumOfDistancesInTree[u] += dpSumOfDistancesInTree[v] + szSumOfDistancesInTree[v];
            szSumOfDistancesInTree[u] += szSumOfDistancesInTree[v];
        }
    }

    private void dfsSumOfDistancesInTree_2(int u,int f){
        ansSumOfDistancesInTree[u] = dpSumOfDistancesInTree[u];
        for(int v:graphSumOfDistancesInTree.get(u)){
            if(v == f){
                continue;
            }
            int pu = dpSumOfDistancesInTree[u],pv = dpSumOfDistancesInTree[v];
            int su = szSumOfDistancesInTree[u],sv = szSumOfDistancesInTree[v];
            dpSumOfDistancesInTree[u] -= dpSumOfDistancesInTree[v] + szSumOfDistancesInTree[v];
            szSumOfDistancesInTree[u] -= szSumOfDistancesInTree[v];
            dpSumOfDistancesInTree[v] += dpSumOfDistancesInTree[u] + szSumOfDistancesInTree[u];
            szSumOfDistancesInTree[v] += szSumOfDistancesInTree[u];

            dfsSumOfDistancesInTree_2(v, u);

            dpSumOfDistancesInTree[u] = pu;
            dpSumOfDistancesInTree[v] = pv;
            szSumOfDistancesInTree[u] = su;
            szSumOfDistancesInTree[v] = sv;

        }
    }

    /**
     * 530. 二叉搜索树的最小绝对差
     *
     * 给你一棵所有节点为非负值的二叉搜索树，请你计算树中任意两节点的差的绝对值的最小值。
     *
     * @param root
     * @return
     */
    int preGetMinimumDifference;
    int ansGetMinimumDifference;
    public int getMinimumDifference(TreeNode root) {
        ansGetMinimumDifference = Integer.MAX_VALUE;
        preGetMinimumDifference = -1;
        dfsGetMinimumDifference(root);
        return ansGetMinimumDifference;
    }
    private void dfsGetMinimumDifference(TreeNode root){
        if(root == null){
            return;
        }
        dfsGetMinimumDifference(root.left);
        if(preGetMinimumDifference == -1){
            preGetMinimumDifference = root.val;
        }else{
            ansGetMinimumDifference = Math.min(ansGetMinimumDifference,root.val - preGetMinimumDifference);
            preGetMinimumDifference = root.val;
        }
        dfsGetMinimumDifference(root.right);
    }

    /**
     *
     * @param root
     * @return
     */
    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if(root == null){
            return res;
        }

        Deque<TreeNode> stack = new LinkedList<>();
        TreeNode node = root;
        while (!stack.isEmpty() || node != null){
            while (node != null){
                res.add(node.val);
                stack.push(node);
                node = node.left;
            }
            node = stack.pop();
            node = node.right;
        }
        return res;
    }

    /**
     * 129. 求根到叶子节点数字之和
     *
     * 给定一个二叉树，它的每个结点都存放一个 0-9 的数字，每条从根到叶子节点的路径都代表一个数字。
     *
     * 例如，从根到叶子节点路径 1->2->3 代表数字 123。
     *
     * 计算从根到叶子节点生成的所有数字之和。
     *
     * 说明: 叶子节点是指没有子节点的节点。
     *
     * @param root
     * @return
     */
    public int sumNumbers(TreeNode root) {
        int result = 0;
        if(root == null){
            return 0;
        }
        Queue<TreeNode> nodeQueue = new LinkedList<>();
        Queue<Integer> numQueue = new LinkedList<Integer>();
        nodeQueue.offer(root);
        numQueue.offer(root.val);
        while (!nodeQueue.isEmpty()){
            TreeNode node = nodeQueue.poll();
            int num = numQueue.poll();
            TreeNode left = node.left,right = node.right;
            if(left == null && right == null){
                result += num;
            }else {
                if(left !=null){
                    nodeQueue.offer(left);
                    numQueue.offer(num * 10 + left.val);
                }
                if(right != null){
                    nodeQueue.offer(right);
                    numQueue.offer(num * 10 + right.val);
                }
            }
        }
        return result;
    }

    /**
     * 222. 完全二叉树的节点个数
     * 给出一个完全二叉树，求出该树的节点个数。
     *
     * 说明：
     *
     * 完全二叉树的定义如下：在完全二叉树中，除了最底层节点可能没填满外，其余每层节点数都达到最大值，并且最下面一层的节点都集中在该层最左边的若干位置。若最底层为第 h 层，则该层包含 1~ 2h 个节点。
     *
     * @param root
     * @return
     */
    public int countNodes(TreeNode root) {
        if(root == null){
            return 0;
        }
        int level = 0;
        TreeNode node = root;
        while (node.left != null){
            level++;
            node = node.left;
        }
        int low = 1 << level, high = (1 << (level + 1)) - 1;
        while(low < high){
            int mid = (high - low + 1) / 2 + low;
            if(existsCountNodes(root,level,mid)){
                low = mid;
            }else{
                high = mid - 1;
            }
        }
        return low;
    }

    private boolean existsCountNodes(TreeNode root,int level,int k){
        int bits = 1 << (level - 1);
        TreeNode node = root;
        while (node != null && bits > 0){
            if((bits & k) == 0){
                node = node.left;
            }else {
                node = node.right;
            }
            bits >>= 1;
        }
        return node != null;
    }
}
