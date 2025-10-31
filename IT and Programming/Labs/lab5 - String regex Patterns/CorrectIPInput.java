import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;

public class CorrectIPInput {

    public static void main(String[] args) {
        // требования к IP
        //1. Четыре числа
        //2. Разделены точками
        //3. Каждое число в диапазоне от 0 до 255
        

        try {
            String input = "112.0.0.0";
            //String passwordPattern = "^((25[0-5]|2[0-4][0-9]|1?(0{1}|[1-9]{1,2}))\\.){3}(25[0-5]|2[0-4][0-9]|1?(0{1}|[0-9]{2}^0{2,3}))$";
            String passwordPattern = "^((25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])\\.){3}(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])$";
            Pattern pattern = Pattern.compile(passwordPattern);
            Matcher matcher = pattern.matcher(input);
            if (matcher.find()) {
                        System.out.printf("Корректный IP");                
                } else{
                    System.out.printf("Введенный IP адрес не корректен");
                }
            
        } catch (PatternSyntaxException e) {
            System.err.println("Ошибка в синтаксисе регулярного выражения: " + e.getMessage());
            System.err.println("Описание ошибки: " + e.getDescription());
            System.err.println("Позиция ошибки: " + e.getIndex());
        } catch (Exception e) {
            System.err.println("Произошла ошибка: " + e.getMessage());
            e.printStackTrace();
        }
    }
}