package algorithm;

import java.util.regex.Matcher;
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

    public static String replaceStrToDate(String content){
        content = content.replaceAll("/", "-");
        content = content.replaceAll("\\.", "-");
        String[] tmp = content.split("-");
        StringBuffer sb = new StringBuffer();
        for(int i = 0; i < tmp.length; i++){
            if(i != 0){
                sb.append("-");
            }
            if(tmp[i].length() == 1){
                sb.append("0");
            }
            sb.append(tmp[i]);

        }
        return sb.toString();
    }

    public static boolean checkDateFormat(String checkValue){
        String eL = "^((\\d{2}(([02468][048])|([13579][26]))[\\-\\/\\s]?((((0?[13578])|(1[02]))[\\-\\/\\s]?((0?[1-9])|([1-2][0-9])|(3[01])))|(((0?[469])|(11))[\\-\\/\\s]?((0?[1-9])|([1-2][0-9])|(30)))|(0?2[\\-\\/\\s]?((0?[1-9])|([1-2][0-9])))))|(\\d{2}(([02468][1235679])|([13579][01345789]))[\\-\\/\\s]?((((0?[13578])|(1[02]))[\\-\\/\\s]?((0?[1-9])|([1-2][0-9])|(3[01])))|(((0?[469])|(11))[\\-\\/\\s]?((0?[1-9])|([1-2][0-9])|(30)))|(0?2[\\-\\/\\s]?((0?[1-9])|(1[0-9])|(2[0-8]))))))";
        Pattern p = Pattern.compile(eL);
        Matcher m = p.matcher(checkValue);
        return m.matches();
    }

}
