import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;

public class CheckCorrectPassword {

    public static void main(String[] args) {
        // требования к паролю
        //1. Латинские буквы и цифры
        //2. длина от 8 до 16 символов
        //3. Содержать хотя бы одну заглавную букву 
        //4. Содержать хотя бы одну цифру

        try {
            String password = "Y8reneggkngn3";
            String passwordPattern = "^(?=.*[A-Z])(?=.*\\d)[A-Za-z0-9]{8,16}$";
            Pattern pattern = Pattern.compile(passwordPattern);
            Matcher matcher = pattern.matcher(password);
            if (matcher.find()) {
                        System.out.printf("Пароль является корректным");                
                } else{
                    System.out.printf("Пароль не прошел проверку");
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