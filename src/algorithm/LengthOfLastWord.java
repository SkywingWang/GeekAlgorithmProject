package algorithm;

public class LengthOfLastWord {
    public int lengthOfLastWord(String s) {
        if (s == null || s.length() == 0)
            return 0;
        int i = s.length() - 1;
        char space = new Character(' ');
        while (i >= 0) {
            if (s.charAt(i) == space) {
                i--;
            } else {
                int j = i;
                while (j >= 0) {
                    if (s.charAt(j) == space) {
                        return i - j;
                    } else {
                        j--;
                    }
                }
                return i - j;
            }
        }
        return 0;
    }
}
