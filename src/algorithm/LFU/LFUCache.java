package algorithm.LFU;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by skywingking
 * on 2020/4/5
 */
public class LFUCache {

    Map<Integer,LFUNode> cache;

    LFUDoubleLinkedList firstLinkedList;

    LFUDoubleLinkedList lastLinkedList;

    int size;

    int capacity;

    public LFUCache(int capacity) {
        cache = new HashMap<>(capacity);
        firstLinkedList = new LFUDoubleLinkedList();
        lastLinkedList = new LFUDoubleLinkedList();
        firstLinkedList.post = lastLinkedList;
        lastLinkedList.pre = firstLinkedList;
        this.capacity = capacity;
    }

    public int get(int key) {
        LFUNode node = cache.get(key);
        if(node == null)
            return -1;
        freqInc(node);
        return node.value;
    }

    public void put(int key, int value) {
        if(capacity == 0)
            return;
        LFUNode node = cache.get(key);
        if(node != null){
            node.value = value;
            freqInc(node);
        }else{
            if(size == capacity){
                cache.remove(lastLinkedList.pre.tail.pre.key);
                lastLinkedList.removeNode(lastLinkedList.pre.tail.pre);
                size --;
                if(lastLinkedList.pre.head.post == lastLinkedList.pre.tail)
                    removeDoubleLinkedList(lastLinkedList.pre);
            }
            LFUNode newNode = new LFUNode(key,value);
            cache.put(key,newNode);
            if(lastLinkedList.pre.freq != 1){
                LFUDoubleLinkedList newDoubleLinkedList = new LFUDoubleLinkedList(1);
                addDoubleLinkedList(newDoubleLinkedList,lastLinkedList.pre);
                newDoubleLinkedList.addNode(newNode);
            }else{
                lastLinkedList.pre.addNode(newNode);
            }
            size++;
        }

    }

    public void freqInc(LFUNode node){
        LFUDoubleLinkedList linkedList = node.lfuDoubleLinkedList;
        LFUDoubleLinkedList preLinkedList = linkedList.pre;
        linkedList.removeNode(node);
        if(linkedList.head.post == linkedList.tail)
            removeDoubleLinkedList(linkedList);
        node.freq ++;
        if(preLinkedList.freq != node.freq){
            LFUDoubleLinkedList newDoubleLinkedList = new LFUDoubleLinkedList(node.freq);
            addDoubleLinkedList(newDoubleLinkedList,preLinkedList);
            newDoubleLinkedList.addNode(node);
        }else{
            preLinkedList.addNode(node);
        }
    }

    private void addDoubleLinkedList(LFUDoubleLinkedList newDoubleLinedList,LFUDoubleLinkedList preLinkedList){
        newDoubleLinedList.post = preLinkedList.post;
        newDoubleLinedList.post.pre = newDoubleLinedList;
        newDoubleLinedList.pre = preLinkedList;
        preLinkedList.post = newDoubleLinedList;
    }

    private void removeDoubleLinkedList(LFUDoubleLinkedList doubleLinkedList){
        doubleLinkedList.pre.post = doubleLinkedList.post;
        doubleLinkedList.post.pre = doubleLinkedList.pre;
    }
}
