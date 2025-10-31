
public class Palindrome {
    public static void main(String[] args){
        for (int i = 0; i < args.length; i++) {
            String s = args[i];
            System.out.println(isPalindrome(s));
            
        }
        
        
    }
    public static String reverseString(String a) {
        String stroka = "";
        for (int i = 0; i < a.length(); i++){
            stroka = a.charAt(i) + stroka;

        }
        return stroka;
    }
    public static boolean isPalindrome(String s) {
        String step1 = reverseString(s);
        boolean step2 = s.equals(step1);
        return step2;
    }
            
}