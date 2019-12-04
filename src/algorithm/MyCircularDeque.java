package algorithm;

import java.util.ArrayDeque;

/**
 * created by Sven
 * on 2019-11-29
 * <p>
 * 设计循环双端队列
 * <p>
 * MyCircularDeque(k)：构造函数,双端队列的大小为k。
 * insertFront()：将一个元素添加到双端队列头部。 如果操作成功返回 true。
 * insertLast()：将一个元素添加到双端队列尾部。如果操作成功返回 true。
 * deleteFront()：从双端队列头部删除一个元素。 如果操作成功返回 true。
 * deleteLast()：从双端队列尾部删除一个元素。如果操作成功返回 true。
 * getFront()：从双端队列头部获得一个元素。如果双端队列为空，返回 -1。
 * getRear()：获得双端队列的最后一个元素。 如果双端队列为空，返回 -1。
 * isEmpty()：检查双端队列是否为空。
 * isFull()：检查双端队列是否满了。
 */
public class MyCircularDeque {
    int[] deque;
    private int head, tail;
    private boolean isEmpty = true;

    /**
     * Initialize your data structure here. Set the size of the deque to be k.
     */
    public MyCircularDeque(int k) {
        deque = new int[k];
        head = tail = 0;
    }

    /**
     * Adds an item at the front of Deque. Return true if the operation is successful.
     */
    public boolean insertFront(int value) {
        if(isEmpty){
            deque[head] = value;
            isEmpty = false;
            return true;
        }
        if (getRealIndex(head - 1) == tail)
            return false;
        else {
            head = getRealIndex(head - 1);
            deque[head] = value;
            if (isEmpty)
                isEmpty = false;
            return true;
        }
    }

    /**
     * Adds an item at the rear of Deque. Return true if the operation is successful.
     */
    public boolean insertLast(int value) {
        if(isEmpty){
            deque[tail] = value;
            isEmpty = false;
            return true;
        }
        if (getRealIndex(tail + 1) == head)
            return false;
        else {
            tail = getRealIndex(tail + 1);
            deque[tail] = value;
            if (isEmpty)
                isEmpty = false;
            return true;
        }
    }

    /**
     * Deletes an item from the front of Deque. Return true if the operation is successful.
     */
    public boolean deleteFront() {
        if(isEmpty)
            return false;
        else if (head == tail){
            isEmpty = true;
            return true;
        }
        else {
            head = getRealIndex(head + 1);
            return true;
        }

    }

    /**
     * Deletes an item from the rear of Deque. Return true if the operation is successful.
     */
    public boolean deleteLast() {
        if(isEmpty)
            return false;
        else if (head == tail){
            isEmpty = true;
            return true;
        }else {
            tail = getRealIndex(tail - 1);
            return true;
        }
    }

    /**
     * Get the front item from the deque.
     */
    public int getFront() {
        if(isEmpty)
            return -1;
        return deque[head];
    }

    /**
     * Get the last item from the deque.
     */
    public int getRear() {
        if(isEmpty)
            return -1;
        return deque[tail];
    }

    /**
     * Checks whether the circular deque is empty or not.
     */
    public boolean isEmpty() {
        return isEmpty;
    }

    /**
     * Checks whether the circular deque is full or not.
     */
    public boolean isFull() {
        if(getRealIndex(tail + 1) == head)
            return true;
        else
            return false;
    }

    private int getRealIndex(int index) {
        int arrayLength = deque.length;
        if (index < 0)
            index += arrayLength;
        else if (index >= arrayLength)
            index -= arrayLength;
        return index;
    }
}
