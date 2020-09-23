package data;

/**
 * Created by skywingking
 * on 2020/9/17
 */
public class UnionFind {
    int[] ancestor;

    public UnionFind(int n){
        ancestor = new int[n];
        for(int i = 0; i < n; i++){
            ancestor[i] = i;
        }
    }
    public void union(int index1,int index2){
        ancestor[find(index1)] = find(index2);
    }

    public int find(int index){
        if(ancestor[index] != index){
            ancestor[index] = find(ancestor[index]);
        }
        return ancestor[index];
    }
}
