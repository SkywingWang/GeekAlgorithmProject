package algorithm;

import data.MapNode;

import java.util.ArrayList;
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
}
