package data;

/**
 * Created by skywingking
 * on 2021/1/18
 */
public class UnionFindAccountMerge {
    int[] parent;

    public UnionFindAccountMerge(int n){
        parent = new int[n];
        for(int i = 0; i < n; i++){
            parent[i] = i;
        }
    }
    public void union(int index1,int index2){
        parent[find(index2)] = find(index1);
    }

    public int find(int index){
        if(parent[index] != index){
            parent[index] = find(parent[index]);
        }
        return parent[index];
    }
}
