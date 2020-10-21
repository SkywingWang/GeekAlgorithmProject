import algorithm.ArrayAlgorithm;
import algorithm.RegularExpressionTest;
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

//        System.out.println("姓名 result : " + RegularExpressionTest.checkName("刘w"));
        System.out.println("日期校验 result ：" + RegularExpressionTest.checkDateFormat("2011-02-28"));

        int a = 13 & 17;
        System.out.println(a);


    }
}
