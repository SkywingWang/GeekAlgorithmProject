import algorithm.ArrayAlgorithm;
import algorithm.RegularExpressionTest;
import algorithm.StringAlgorithm;
import algorithm.VariousAlgorithm;

public class Main {

    public static void main(String[] args) {
        int i = 2001;
        int j = 1;
        int[] test = {2,1,4,7,3,2,5};
        int[] testMountain = {0,3,2,1};
        ArrayAlgorithm arrayAlgorithm = new ArrayAlgorithm();
        System.out.println(arrayAlgorithm.longestMountain(test));
        System.out.println(arrayAlgorithm.validMountainArray(testMountain));
    }
}
