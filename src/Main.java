import algorithm.ArrayAlgorithm;
import algorithm.RegularExpressionTest;
import algorithm.StringAlgorithm;
import algorithm.VariousAlgorithm;

import java.text.SimpleDateFormat;
import java.util.*;

public class Main {

    public static void main(String[] args) {
        String content = "2021-01-13 02:10:26,392 - SysLogger_logger - INFO - <Arilou Header> BL_c0_CEF_report.txt <-> <-> CEF:0|Arilou|Sentinel-ETH|1.1.1|10|FilterRule|0|aid=1 deviceExternalId=AR1LU0VEH1CLE0001 requestContext=BL_c0 reason=HighReliability externalId=1 cnt=1 spid=1 dvcpid=0 proto=DHCP eventOutcome=Allowed msg=ffffffffffff02000000000181000001080045000114d56b40002011846e00000000ffffffff004400430100a63d010106005b0a00000000000000000000000000000000000000000000020000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000006382536335010137020103ff";
        String logDate = content.substring(0,19);
        content = content.substring(content.indexOf("CEF:") + 4);
        int splitSpaceIndex = content.indexOf(" ");
        String cef = content.substring(0,splitSpaceIndex);
        String[] cefArr = cef.split("\\|");
        String version = cefArr[0];
        String deviceVendor = cefArr[1];
        String deviceProduct = cefArr[2];
        String deviceVersion = cefArr[3];
        String eventName = cefArr[4];
        String security = cefArr[5];
        String extension = content.substring(splitSpaceIndex);
        String result = logDate + " - "  + version + " - " + deviceVendor + " - " + deviceProduct + " - " + deviceVersion + " - " + eventName + " - " + security +" - " + extension;
        System.out.println(cef);
        System.out.println(result);
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
