package data;

/**
 * Created by skywingking
 * on 2021/1/16
 */
public class UnionFindHitBricks {
    private int[] parent;

    private int[] size;

    public UnionFindHitBricks(int n){
        parent = new int[n];
        size = new int[n];
        for(int i = 0; i < n; i ++){
            parent[i] = i;
            size[i] = 1;
        }
    }

    public int find(int x){
        if(x != parent[x]){
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }

    public void union(int x,int y){
        int rootX = find(x);
        int rootY = find(y);

        if (rootX == rootY) {
            return;
        }
        parent[rootX] = rootY;
        // 在合并的时候维护数组 size
        size[rootY] += size[rootX];
    }

    public int getSize(int x){
        int root = find(x);
        return size[root];
    }
}
