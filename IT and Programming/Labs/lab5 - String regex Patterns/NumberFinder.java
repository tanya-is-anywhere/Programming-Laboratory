import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;

public class NumberFinder {

    public static void main(String[] args) {
        
        try {
            String text = "The price of -25 the product is $19.99";
            Pattern pattern = Pattern.compile("(-)?\\d+(\\.\\d+)?");
            Matcher matcher = pattern.matcher(text);
            while (matcher.find()) {
                System.out.println(matcher.group());
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

