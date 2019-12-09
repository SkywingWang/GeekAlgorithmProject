import algorithm.RotateArray;
import algorithm.XSqrt;

public class Main {

    public static void main(String[] args) {

        RotateArray rotateArray = new RotateArray();
        int[] test = {1,2,3,4,5,6,7};
        rotateArray.rotate(test,3);

        XSqrt xSqrt = new XSqrt();
        System.out.println(Integer.MAX_VALUE + "");
        System.out.println(xSqrt.mySqrt(2147395600));
    }
}
