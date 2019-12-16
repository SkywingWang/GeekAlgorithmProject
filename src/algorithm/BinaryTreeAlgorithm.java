package algorithm;


import data.TreeNode;

import java.util.ArrayList;
import java.util.List;

/**
 * created by Sven
 * on 2019-12-14
 *
 * 二叉树的相关算法
 *
 */
public class BinaryTreeAlgorithm {

    /**
     * created by Sven
     * on 2019-12-12
     *
     * 相同的树
     *
     * 给定两个二叉树，编写一个函数来检验它们是否相同。
     *
     * 如果两个树在结构上相同，并且节点具有相同的值，则认为它们是相同的。
     */
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if(p == null && q == null)
            return true;
        if(p == null || q == null)
            return false;
        if(p.val != q.val)
            return false;
        return isSameTree(p.left,q.left) && isSameTree(p.right,q.right);
    }


    /**
     * created by Sven
     * on 2019-12-12
     *
     * 镜像树 对称二叉树
     *
     * 给定一个二叉树，检查它是否是镜像对称的。
     *
     * 例如，二叉树 [1,2,2,3,4,4,3] 是对称的。
     */
    public boolean isSymmetric(TreeNode root){
        return isMirror(root,root);
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
     *
     * 二叉树的最大深度
     *
     * 给定一个二叉树，找出其最大深度。
     *
     * 二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。
     *
     * 说明: 叶子节点是指没有子节点的节点。
     */
    public int maxDepth(TreeNode root) {
        if(root == null)
            return 0;
        if(root.left == null && root.right == null)
            return 1;
        return Math.max(maxDepth(root.left),maxDepth(root.right)) + 1;
    }

    /**
     * created by Sven
     * on 2019-12-14
     *
     * 二叉树的层次遍历
     *
     * 给定一个二叉树，返回其节点值自底向上的层次遍历。 （即按从叶子节点所在层到根节点所在的层，逐层从左向右遍历）
     */
    public List<List<Integer>> levelOrderBottom(TreeNode root){
        List<List<Integer>> ans = new ArrayList<>();
        DFS(root,0,ans);
        return ans;
    }

    private void DFS(TreeNode root,int level,List<List<Integer>> ans){
        if(root == null){
            return;
        }

        //当前层如果还没有元素，先new一个空列表
        if(ans.size() <= level){
            ans.add(0,new ArrayList<>());
        }
        ans.get(ans.size() - 1 - level).add(root.val);
        DFS(root.left,level + 1,ans);
        DFS(root.right,level + 1,ans);
    }
}
