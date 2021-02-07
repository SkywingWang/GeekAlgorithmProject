package algorithm;

import data.MapNode;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

/**
 * Created by skywingking
 * on 2020/8/12
 */
public class MapAlgorithm {

    /**
     * 133. 克隆图
     *
     * 给你无向 连通 图中一个节点的引用，请你返回该图的 深拷贝（克隆）。
     *
     * 图中的每个节点都包含它的值 val（int） 和其邻居的列表（list[Node]）。
     * @param node
     * @return
     */
    private HashMap<MapNode,MapNode> visited = new HashMap<>();
    public MapNode cloneGraph(MapNode node) {
        if(node == null)
            return node;
        if(visited.containsKey(node))
            return visited.get(node);
        MapNode cloneNode = new MapNode(node.val,new ArrayList<>());
        visited.put(node,cloneNode);
        for(MapNode neighbor : node.neighbors){
            cloneNode.neighbors.add(cloneGraph(neighbor));
        }
        return cloneNode;
    }

    /**
     * 1579. 保证图可完全遍历
     *
     * Alice 和 Bob 共有一个无向图，其中包含 n 个节点和 3  种类型的边：
     *
     * 类型 1：只能由 Alice 遍历。
     * 类型 2：只能由 Bob 遍历。
     * 类型 3：Alice 和 Bob 都可以遍历。
     * 给你一个数组 edges ，其中 edges[i] = [typei, ui, vi] 表示节点 ui 和 vi 之间存在类型为 typei 的双向边。请你在保证图仍能够被 Alice和 Bob 完全遍历的前提下，找出可以删除的最大边数。如果从任何节点开始，Alice 和 Bob 都可以到达所有其他节点，则认为图是可以完全遍历的。
     *
     * 返回可以删除的最大边数，如果 Alice 和 Bob 无法完全遍历图，则返回 -1 。
     *
     * @param n
     * @param edges
     * @return
     */
    public int maxNumEdgesToRemove(int n, int[][] edges) {
        UnionFindMNETR ufa = new UnionFindMNETR(n);
        UnionFindMNETR ufb = new UnionFindMNETR(n);
        int ans = 0;
        for(int[] edge : edges){
            --edge[1];
            --edge[2];
        }

        for(int[] edge:edges){
            if(edge[0] == 3){
                if(!ufa.unite(edge[1],edge[2])){
                    ++ans;
                }else{
                    ufb.unite(edge[1],edge[2]);
                }
            }
        }

        for(int[] edge:edges){
            if(edge[0] == 1){
                if(!ufa.unite(edge[1],edge[2])){
                    ++ans;
                }
            }else if(edge[0] == 2){
                if(!ufb.unite(edge[1],edge[2])){
                    ++ans;
                }
            }
        }

        if(ufa.setCount != 1 || ufb.setCount != 1){
            return -1;
        }
        return ans;
    }

    private class UnionFindMNETR{
        int [] parent;
        int [] size;
        int n;
        int setCount;

        public UnionFindMNETR(int n){
            this.n = n;
            this.setCount = n;
            this.parent = new int[n];
            this.size = new int[n];
            Arrays.fill(size,1);
            for(int i = 0; i < n; i++){
                parent[i] = i;
            }
        }

        public int findset(int x){
            return parent[x] == x ? x : (parent[x] = findset(parent[x]));
        }

        public boolean unite(int x,int y){
            x = findset(x);
            y = findset(y);
            if(x == y){
                return false;
            }
            if(size[x] < size[y]){
                int temp = x;
                x = y;
                y = temp;
            }
            parent[y] = x;
            size[x] += size[y];
            --setCount;
            return true;
        }

        public boolean connected(int x, int y){
            x = findset(x);
            y = findset(y);
            return x == y;
        }
    }


}
