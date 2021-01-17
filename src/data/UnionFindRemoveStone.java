package data;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by skywingking
 * on 2021/1/15
 */
public class UnionFindRemoveStone {
    private Map<Integer,Integer> parent;
    private int count;

    public UnionFindRemoveStone(){
        this.parent = new HashMap<>();
        this.count = 0;
    }

    public int getCount(){
        return count;
    }

    public int find(int x){
        if(!parent.containsKey(x)){
            parent.put(x,x);
            count++;
        }
        if( x != parent.get(x)){
            parent.put(x,find(parent.get(x)));
        }
        return parent.get(x);
    }

    public void union(int x,int y){
        int rootX = find(x);
        int rootY = find(y);
        if(rootX == rootY){
            return;
        }
        parent.put(rootX,rootY);
        count--;
    }
}
