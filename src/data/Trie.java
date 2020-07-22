package data;

/**
 * Created by skywingking
 * on 2020/7/9
 */
public class Trie {
    public Trie[] next;
    public boolean isEnd;

    public Trie(){
        next = new Trie[26];
        isEnd = false;
    }

    public void insert(String s){
        Trie curPos = this;
        for(int i = s.length() - 1; i >= 0; i--){
            int t = s.charAt(i) - 'a';
            if(curPos.next[t] == null)
                curPos.next[t] = new Trie();
            curPos = curPos.next[t];
        }
        curPos.isEnd = true;
    }
}
