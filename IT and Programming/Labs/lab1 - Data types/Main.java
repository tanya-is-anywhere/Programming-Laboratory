/*public class Palindrome {
    public static void main(String[] args){
        for (int i = 0; i < args.length; i++) {
            String s = args[i];
            
        }
        
        
    }
    public static String reverseString(String a) {
        String stroka = "";
        for (int i = 0; i < a.length; i++){
            stroka = a.charAt(i) + stroka;

        }
    }
    public static String isPalindrome(String a) {
        step1 = reverseString(s);
        step2 = s.equals(step1);
        return step2;
    }
            
}*/

public class Main {
    public static void main(String[] args){
        /*info("Henry");
        String java = "Java";
        info(java);
        info("");*/

        info(String.valueOf(summa((short)5, (short)7)));
    }
    public static int summa(short a, short b){
        int res = a + b;
        //System.out.println("Результат: " + res);
        return res;
    }
    public static void info(String word) {
        // тело функции
        System.out.print("Hello " + word);
        System.out.println("!");
    }
    
}