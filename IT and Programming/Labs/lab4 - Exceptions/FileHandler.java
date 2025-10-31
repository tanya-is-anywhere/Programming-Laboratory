import java.io.*;
// вариант 2
public class FileHandler {
    public static void main(String[] args) {
        File source = new File("D:/otherfiles/ITandP/Labs/lab4/file.txt");
        File target = new File("D:/otherfiles/ITandP/Labs/lab4/file2.txt");
        
        try (FileReader reader = new FileReader(source);
             FileWriter writer = new FileWriter(target)) {
            
            int character;
            while ((character = reader.read()) != -1) {
                try {
                    reader.close();
                    writer.write(character);
                } catch (IOException e) {
                    System.err.println("Ошибка при записи символа: " + e.getMessage());
                }
            }
            
        } catch (FileNotFoundException e) {
            System.err.println("Файл не найден: " + e.getMessage());
        } catch (IOException e) {
            System.err.println("Ошибка при чтении файла: " + e.getMessage());
        } finally {
            System.out.println("Операция завершена");
        }
    }
}
