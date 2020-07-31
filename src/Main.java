import algorithm.ArrayAlgorithm;
import algorithm.StringAlgorithm;
import algorithm.VariousAlgorithm;

public class Main {

    public static void main(String[] args) {
        int i = 2001;
        int j = 1;
        int[] test = {1,2,3,4};
        System.out.println("result : " + ((i & j) == j));

        StringAlgorithm stringAlgorithm = new StringAlgorithm();
        System.out.println(stringAlgorithm.minWindow("a","a"));
        ArrayAlgorithm arrayAlgorithm = new ArrayAlgorithm();
        arrayAlgorithm.productExceptSelf(test);

        System.out.println("result : " + stringAlgorithm.isSubsequence("abc","ahbgdc"));
        stringAlgorithm.testStr("https://nngcockpit.oss-ap-southeast-1.aliyuncs.com/cockpit/rp/2a0382f3-9fe1-4bbf-82f8-70cd24301a6f.jpg");
    }
}
