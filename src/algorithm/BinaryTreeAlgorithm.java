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
}
