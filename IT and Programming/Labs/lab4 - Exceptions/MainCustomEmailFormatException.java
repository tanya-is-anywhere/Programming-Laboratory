
/*public class CustomEmailFormatException {
    /*Реализуйте программу, которая проверяет формат email- 
адреса с использованием этого класса, и, если адрес не соответ- 
ствует формату, выбрасывайте исключение CustomEmailFormat
Exception.
    public void InnerCustomEmailFormatException(File file) {
        // обработка входных данных - чтение формата email-адреса
        // выделение исключений - CustomEmailFormatException
        // запись исключений в текстовый файл
    }
    public static void main(String[] args) {
        
    }
}*/


/* 
import java.util.regex.Pattern;
import CustomEmailFormatException;
public class MainCustomEmailFormatException {

    public static void main(String[] args) {
        CustomEmailFormatException validator = new CustomEmailFormatException();
        
        try {
            // Тестовые email адреса
            String[] testEmails = {
                "test@example.com",
                "invalid-email",
                "another.test@domain.co.uk",
                "wrong@domain",
                "correct@domain.com"
            };
            
            for (String email : testEmails) {
                try {
                    if (validator.chechFormat(email)) {
                        System.out.println(email + " - корректный email");
                    }
                } catch (CustomEmailFormatException e) {
                    System.out.println(email + ": " + e.getMessage());
                }
            }
        } catch (Exception e) {
            System.err.println("Произошла ошибка: " + e.getMessage());
    }
}

}*/

import java.CustomEmailFormatException;

// Основной класс для тестирования
public class MainCustomEmailFormatException {
    public static void main(String[] args) {
        try {
            // Пример использования исключения
            String email = "invalid-email";
            if (!isValidEmail(email)) {
                throw new CustomEmailFormatException("Неверный формат email: " + email);
            }
        } catch (CustomEmailFormatException e) {
            System.err.println(e.getMessage());
        }
    }
    
    private static boolean isValidEmail(String email) {
        // Простая проверка email
        return email.contains("@");
    }
}
