package algorithm;


import data.TreeNode;
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
}
