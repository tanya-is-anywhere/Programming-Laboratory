import java.util.LinkedList;  
import java.util.List;  

public class Shop {
    public static class HashTable<K, V> {  
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
    }  

    public void search(K key){
        Order order = (Order) get(key);
            if (order != null) {
                order.info();
            } else {
                System.out.println("Заказ не найден");
            }
    }

    }
    public static class Order {
        private int id;
        private String orderDate;
        private String[] items;  
        private boolean status;
        public Order(int id, String orderDate, String[] items, boolean status){
            this.id = id;
            this.orderDate = orderDate;
            this.items = items;
            this.status = status; // пусть 1 значит заказ выполнен, 0 - не выполнен
            
        }
        public void changeStatus(boolean status){
            this.status = status;
        }
        public int getId(){
            return this.id;
        }
        private void info(){
            String orderStatus;
            if (status == false){
                orderStatus = "В процессе выполнения";
            }
            else{
                orderStatus = "Завершен";
            }
            System.out.println("ID заказа: "+id);
            System.out.println("Дата заказа: "+orderDate);
            System.out.println("Товары: ");
            for (int i=0; i < items.length; i++){
                System.out.print(items[i]+", ");
            }
            System.out.println("");
            System.out.println("Статус: "+orderStatus);
        }
    }

    public static void main(String[] args){
            HashTable<Integer, Order> table1 = new HashTable<>(5);
            Order order1 = new Order(1, "07/10/2025", new String[] {"Товар 1", "Товар 2", "Товар 3"}, false);
            Order order2 = new Order(1, "07/10/2025", new String[] {"Товар 1", "Товар 2", "Товар 3"}, false);


            // операция вставки
            table1.put(order1.getId(), order1);
            table1.put(order2.getId(), order1);
            // операция поиска 
            table1.search(order1.getId());
            // операция удаления заказа по номеру (после - проверка, нужно попробовать найти его снова)
            table1.remove(order1.getId());
            table1.search(order1.getId());
            
        }

}