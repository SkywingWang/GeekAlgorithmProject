package algorithm.LFU;

import java.util.HashMap;
import java.util.Map;

/**
 * 460 LFU缓存
 *
 * Created by skywingking
 * on 2020/4/5
 *
 * 设计并实现最不经常使用（LFU）缓存的数据结构。它应该支持以下操作：get 和 put。
 *
 * get(key) - 如果键存在于缓存中，则获取键的值（总是正数），否则返回 -1。
 * put(key, value) - 如果键不存在，请设置或插入值。当缓存达到其容量时，它应该在插入新项目之前，使最不经常使用的项目无效。在此问题中，当存在平局（即两个或更多个键具有相同使用频率）时，最近最少使用的键将被去除。
 *
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
