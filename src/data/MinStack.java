package data;

import java.util.Stack;

/**
 * 最小栈
 *
 * 设计一个支持 push，pop，top 操作，并能在常数时间内检索到最小元素的栈。
 *
 * push(x) -- 将元素 x 推入栈中。
 * pop() -- 删除栈顶的元素。
 * top() -- 获取栈顶元素。
 * getMin() -- 检索栈中的最小元素。
 *
 */
public class MinStack {
    /** initialize your data structure here. */

    // 数据栈
    private Stack<Integer> data;

    // 辅助栈
    private Stack<Integer> helper;

    public MinStack() {
        // 数据栈
        data = new Stack<>();
        // 辅助栈
        helper = new Stack<>();
    }

    public void push(int x) {
        data.add(x);
        if(helper.isEmpty() || helper.peek() >= x){
            helper.add(x);
        }else{
            helper.add(helper.peek());
        }
    }

    public void pop() {
        if(!data.isEmpty()){
            helper.pop();
            data.pop();
        }
    }

    public int top() {
        if(!data.isEmpty()){
            return data.peek();
        }throw new RuntimeException("栈中元素为空，此操作非法");
    }

    public int getMin() {
        if(!helper.isEmpty()){
            return helper.peek();
        }throw new RuntimeException("栈中元素为空，此操作非法");
    }
}
