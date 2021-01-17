package data;

/**
 * Created by skywingking
 * on 2021/1/11
 */
public class UnionFindStrSwap {
    private int[] parent;

    private int[] rank;

    public UnionFindStrSwap(int n){
        this.parent = new int[n];
        this.rank = new int[n];
        for(int i = 0; i < n;i++){
            this.parent[i] = i;
            this.rank[i] = 1;
        }
    }

    public void union(int x,int y){
        int rootX = find(x);
        int rootY = find(y);
        if (rootX == rootY) {
            return;
        }

        if (rank[rootX] == rank[rootY]) {
            parent[rootX] = rootY;
            // 此时以 rootY 为根结点的树的高度仅加了 1
            rank[rootY]++;
        } else if (rank[rootX] < rank[rootY]) {
            parent[rootX] = rootY;
            // 此时以 rootY 为根结点的树的高度不变
        } else {
            // 同理，此时以 rootX 为根结点的树的高度不变
            parent[rootY] = rootX;
        }
    }

    public int find(int x) {
        if (x != parent[x]) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }
}
