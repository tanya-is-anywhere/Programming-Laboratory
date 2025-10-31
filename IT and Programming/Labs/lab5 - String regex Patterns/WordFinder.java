import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;

public class WordFinder {

    public static void main(String[] args) {
        // требования
        //1. строка с заданной буквой
        //2. паттерн, проверяющий, что слово начинается с указанной буквы
        //3. вывод слов
        // ? а имеет ли значение регистр
        

        try {
            String input = "The price oF 25 the pRoduct Pis $19.99";
            String letter = "p";
            String letterUp = letter.toUpperCase();
            String letterLower = letter.toLowerCase();
            String passwordPattern = "(" + letterUp + "|" + letterLower + ")[A-Za-z]*";
            Pattern pattern = Pattern.compile(passwordPattern);
            Matcher matcher = pattern.matcher(input);
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