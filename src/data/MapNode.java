package data;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by skywingking
 * on 2020/8/12
 */
public class MapNode {
    public int val;
    public List<MapNode> neighbors;

    public MapNode(){
        val = 0;
        neighbors = new ArrayList<MapNode>();
    }

    public MapNode(int _val){
        val = _val;
        neighbors = new ArrayList<>();
    }

    public MapNode(int _val,ArrayList<MapNode> _neighbors){
        val = _val;
        neighbors = _neighbors;
    }

}
