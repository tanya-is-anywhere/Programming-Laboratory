
// Пользовательское исключение для некорректного формата email

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

class CustomEmailFormatException extends Exception {
    // Конструктор с сообщением об ошибке
    public CustomEmailFormatException(String message) {
        super(message);
    }
}  

public class EmailValidator {
    // Метод для проверки корректности email
    public static void validateEmail(String email) throws CustomEmailFormatException {
        // Регулярное выражение для проверки email
        String emailRegex = "^[A-Za-z0-9+_.-]+@[A-Za-z0-9-]+\\.[A-Za-z]{2,6}$";
        
        // Проверка соответствия email регулярному выражению
        if (!email.matches(emailRegex)) {
            throw new CustomEmailFormatException("Неверный формат email: " + email);
        }
    }
    
    public static void logError(String message) {
        File target = new File("D:/otherfiles/ITandP/Labs/lab4/email_validation.log");
        
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(target, true))) {
            // Добавляем временную метку
            String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
            writer.write(timestamp + " - " + message + "\n");
        } catch (IOException e) {
            System.err.println("Ошибка при записи в лог: " + e.getMessage());
        }
    }
    
    public static void main(String[] args) {
        // Тестовые email адреса
        String[] testEmails = {
            "test@example.com",
            "invalid-email",
            "another.test@domain.co.uk",
            "wrong@domain",
            "correct@domain.com"
        };
        
        // Проверка каждого email
        for (String email : testEmails) {
            try {
                validateEmail(email);
                System.out.println(email + " - корректный email");
            } catch (CustomEmailFormatException e) {
                System.err.println(e.getMessage());
                logError(e.getMessage());
            }
        }
    }
}
