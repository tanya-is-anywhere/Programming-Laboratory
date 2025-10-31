import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;

public class UpperAfterLower {

    public static void main(String[] args) {
        try {
            String text = "The price oF 25 the pRoduct is $19.99";
            Pattern pattern = Pattern.compile("[a-z][A-Z]");
            Matcher matcher = pattern.matcher(text);
    
            // Используем StringBuffer для накопления изменений
            StringBuffer result = new StringBuffer();
    
            while (matcher.find()) {
                // Добавляем замену в StringBuffer
                matcher.appendReplacement(result, "!$0!");
            }
    
            // Добавляем оставшуюся часть строки
            matcher.appendTail(result);
    
            System.out.println(result.toString());
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