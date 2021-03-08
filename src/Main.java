import algorithm.ArrayAlgorithm;
import algorithm.RegularExpressionTest;
import algorithm.StringAlgorithm;
import algorithm.VariousAlgorithm;

import java.text.SimpleDateFormat;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class Main {

    public static void main(String[] args) {

    }


    public static long getTodayStartTime(){
        Calendar calendar = Calendar.getInstance();
        calendar.setTime(new Date());
        calendar.set(Calendar.HOUR_OF_DAY, 0);
        calendar.set(Calendar.MINUTE, 0);
        calendar.set(Calendar.SECOND, 0);
        return calendar.getTime().getTime();
    }

    private static String findJsonValue(String key,String json){
        String regex = "\"" + key + "\": (\"(.*?)\"|(\\d*))";
        Matcher matcher = Pattern.compile(regex).matcher(json);
        String value = null;
        if (matcher.find()) {
            value = matcher.group().split("\\:")[1].replace("\"", "").trim();
            System.out.println(value);
        }
        return value;
    }
}
