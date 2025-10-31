import java.util.LinkedList;  
import java.util.List;  
  
public class HashTable<K, V> {  
    private static int numberElements = 0;
    // Внутренний класс для хранения пар ключ-значение  
    private static class Entry<K, V> {  
        private final K key;  
        private V value;  
        
        public Entry(K key, V value) {  
            this.key = key;  
            this.value = value;  
        }  
          
        public K getKey() {  
            return key;  
        }  
          
        public V getValue() {  
            return value;  
        }  

        public void setValue(V value) {
            this.value = value;
        }
    }  
      
    private final List<List<Entry<K, V>>> table;  
    private final int capacity;  

    public HashTable(int capacity) {  
        this.capacity = capacity;  
        this.table = new LinkedList<>();  
          
        // Инициализируем пустые списки для каждой ячейки  
        for (int i = 0; i < capacity; i++) {  
            table.add(new LinkedList<>());  
        }  
    }  
      
    // Метод для вычисления хэша  
    private int hash(K key) {  
        return Math.abs(key.hashCode()) % capacity;  
    }  
      
    // Добавление пары ключ-значение  
    public void put(K key, V value) {  
        int index = hash(key);  
        List<Entry<K, V>> bucket = table.get(index);  
          
        // Проверяем, существует ли уже такой ключ  
        for (Entry<K, V> entry : bucket) {  
            if (entry.getKey().equals(key)) {  
                entry.setValue(value); // Обновляем значение  
                return;  
            }  
        }  
          
        // Добавляем новую пару  
        bucket.add(new Entry<>(key, value));  
        numberElements++;
    }  
      
    // Получение значения по ключу  
    public V get(K key) {  
        int index = hash(key);  
        List<Entry<K, V>> bucket = table.get(index);  
          
        for (Entry<K, V> entry : bucket) {  
            if (entry.getKey().equals(key)) {  
                return entry.getValue();  
            }  
        }  
          
        return null; // Ключ не найден  
    }  
      
    // Удаление пары по ключу  
    public void remove(K key) {  
        int index = hash(key);  
        List<Entry<K, V>> bucket = table.get(index);  
          
        bucket.removeIf(entry -> entry.getKey().equals(key));  
        numberElements--;
    }  

    // Получение количества элементов в таблице
    public int size(){
        return numberElements;
    }
    // Проверка: является ли таблица пустой
    public boolean isEmpty(){
        if (numberElements == 0){
            return true;
        }
        else {
            return false;
        }
    }


    public static void main(String[] args){
            HashTable table1 = new HashTable<>(5);
            System.out.println(table1.isEmpty());
            table1.put("banana", 125);
            table1.put("kiwi", 180);
            table1.put("apple", 85);
            table1.put("orange", 142);
            System.out.println(table1.get("banana"));
            System.out.println(table1.get("apple"));
            System.out.println(table1.table);
            table1.remove("apple");
            System.out.println(table1.table);
            System.out.println(Math.abs("apple".hashCode()%table1.capacity));
            System.out.println(Math.abs("orange".hashCode()%table1.capacity));
            System.out.println(table1.size());
            System.out.println(table1.isEmpty());
            
            
            
        }
}
