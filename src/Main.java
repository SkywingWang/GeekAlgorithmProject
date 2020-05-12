import algorithm.ArrayAlgorithm;
import algorithm.VariousAlgorithm;

public class Main {

    public static void main(String[] args) {
        VariousAlgorithm variousAlgorithm = new VariousAlgorithm();
        for(int i = 0; i < 7; i++){
            System.out.println(variousAlgorithm.randomStr(7));
        }
        ArrayAlgorithm arrayAlgorithm = new ArrayAlgorithm();
        int[] coins = {2,3,10};
        System.out.println("result : " + arrayAlgorithm.minCount(coins));
    }
}
