package algorithm.LFU;

/**
 * Created by skywingking
 * on 2020/4/5
 */
public class LFUDoubleLinkedList {
    int freq;
    LFUDoubleLinkedList pre;
    LFUDoubleLinkedList post;
    LFUNode head;
    LFUNode tail;

    public LFUDoubleLinkedList(){
        head = new LFUNode();
        tail = new LFUNode();
        head.post = tail;
        tail.pre = head;
    }

    public LFUDoubleLinkedList(int freq){
        head = new LFUNode();
        tail = new LFUNode();
        head.post = tail;
        tail.pre = head;
        this.freq = freq;
    }

    public void removeNode(LFUNode node){
        node.pre.post = node.post;
        node.post.pre = node.pre;
    }

    public void addNode(LFUNode node){
        node.post = head.post;
        head.post.pre = node;
        head.post = node;
        node.pre = head;
        node.lfuDoubleLinkedList = this;
    }
}
