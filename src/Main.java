import algorithm.ArrayAlgorithm;
import algorithm.StringAlgorithm;
import algorithm.VariousAlgorithm;

public class Main {

    public static void main(String[] args) {
        int i = 2001;
        int j = 1;
        System.out.println("result : " + ((i & j) == j));

        StringAlgorithm stringAlgorithm = new StringAlgorithm();
        System.out.println(stringAlgorithm.minWindow("a","a"));
    }
}
