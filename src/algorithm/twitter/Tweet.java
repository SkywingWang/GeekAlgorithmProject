package algorithm.twitter;

/**
 * Created by skywingking
 * on 2020/4/13
 * 推文类，是一个单链表（结点视角）
 */
public class Tweet {
    /**
     * 推文 id
     */
    int id;

    /**
     * 发推文的时间戳
     */
    int timestamp;
    Tweet next;

    public Tweet(int id, int timestamp) {
        this.id = id;
        this.timestamp = timestamp;
    }
}
