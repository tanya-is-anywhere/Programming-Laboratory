import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;
public class TopWords {

    public static void main(String[] args) {
        String filePath = "D:/otherfiles/ITandP/Labs/lab6/text_test.txt";

        File file = new File(filePath);

        try (Scanner scanner = new Scanner(file)) {
            // создаем объект Map для хранения слов и их количества
            Map<String, Integer> wordCount = new HashMap<>();
            while (scanner.hasNext()){
                String word = scanner.next().replaceAll("[^a-zA-Z]", "").toLowerCase();
                if (!word.isEmpty()) {
                    wordCount.put(word, wordCount.getOrDefault(word, 0) + 1);
                }
            }
            // создаем список из элементов Map
            List<Map.Entry<String, Integer>> list = new ArrayList<>(wordCount.entrySet());
            // сортируем список по убыванию количества повторений
            Collections.sort(list, new Comparator<Map.Entry<String,Integer>> (){
                @Override
                public int compare(Map.Entry<String, Integer> o1, Map.Entry<String, Integer> o2) {
                    return o2.getValue().compareTo(o1.getValue());
                }
            });

            // выводим топ-10 слов
            System.out.println("Топ-10 самых частых слов:");
            // выводим результат
            int count = 0;
            for (Map.Entry<String, Integer> entry : list) {
                if (count >= 10) break;
                System.out.printf("%d. %s - %d раз%n",
                        count + 1, entry.getKey(), entry.getValue());
                count++;
            }

        } catch (FileNotFoundException e) {
            System.err.println("Файл не найден: " + filePath);
            e.printStackTrace();
            return;
            
        } catch (Exception e) {
            e.printStackTrace();
        }

        
    }
}