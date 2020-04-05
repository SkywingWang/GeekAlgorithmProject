package algorithm.LFU;

/**
 * Created by skywingking
 * on 2020/4/5
 */
public class LFUNode {
    int key;
    int value;
    int freq = 1;
    LFUNode pre;
    LFUNode post;
    LFUDoubleLinkedList lfuDoubleLinkedList;

    public LFUNode(){}

    public LFUNode(int key,int value){
        this.key = key;
        this.value = value;
    }
}
