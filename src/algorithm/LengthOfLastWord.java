package algorithm;

public class LengthOfLastWord {
    public int lengthOfLastWord(String s) {
        if (s == null || s.length() == 0)
            return 0;
        int i = s.length() - 1;
        while (i >= 0) {
            if (s.charAt(i) == new Character(' ')) {
                i--;
            } else {
                int j = i;
                while (j >= 0) {
                    if (s.charAt(j) == new Character(' ')) {
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
