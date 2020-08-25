package algorithm;

import java.util.regex.Pattern;

/**
 * Created by skywingking
 * on 2020/8/18
 */
public class RegularExpressionTest {
    private static final String  CHINESE_XINJIANG_PATTERN = "^[\u4e00-\u9fa5.·\u36c3\u4DAE]{0,}$";

    public static boolean checkName(String content){
        return Pattern.matches(CHINESE_XINJIANG_PATTERN,content);
    }

    public static String replaceStr(String content){
        return content.replaceAll("//", "-").replaceAll("/.", "-");
    }
}
