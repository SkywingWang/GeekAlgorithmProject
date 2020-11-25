import algorithm.ArrayAlgorithm;
import algorithm.RegularExpressionTest;
import algorithm.StringAlgorithm;
import algorithm.VariousAlgorithm;

import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;

public class Main {

    public static void main(String[] args) {
        int i = 2001;
        int j = 1;
        int[] test = {2,1,4,7,3,2,5};
        int[] testMountain = {0,3,2,1};
        ArrayAlgorithm arrayAlgorithm = new ArrayAlgorithm();
        System.out.println(arrayAlgorithm.longestMountain(test));
        System.out.println(arrayAlgorithm.validMountainArray(testMountain));
        long current = Calendar.getInstance().get(Calendar.HOUR_OF_DAY) * 3600 + Calendar.getInstance().get(Calendar.MINUTE) * 60 +  Calendar.getInstance().get(Calendar.SECOND);
        System.out.println(current);
        long timeStamp = getTodayStartTime();
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd");
        System.out.println(sdf.format(new Date(timeStamp)));
        System.out.println(System.currentTimeMillis());
    }

    public static long getTodayStartTime(){
        Calendar calendar = Calendar.getInstance();
        calendar.setTime(new Date());
        calendar.set(Calendar.HOUR_OF_DAY, 0);
        calendar.set(Calendar.MINUTE, 0);
        calendar.set(Calendar.SECOND, 0);
        return calendar.getTime().getTime();
    }
}
