//import java.util.LinkedList;
//import java.util.Hashtable;
/*public class HashTable {
    
    public static void main(String[] args){
            Hashtable<String, Integer> ht = new Hashtable<>();
            ht.put("One ", 1);
            ht.put("Two ", 2);
            ht.put("Three ", 3);
            System.out.println("Hashtable Elements: " + ht);
        }*/
    
  /*      public void put(K key, V value) {
        int index = hash(key);
        if (table[index] == null) {
            table[index] = new LinkedList< Entry< K, V> > ();
        }
        for (Entry< K, V> entry : table[index]) {
            if (entry.getKey().equals(key)) {
                entry.setValue(value);
                return;
            }
        }
        table[index].add(new Entry< K, V> (key, value));
        size++;
        }
    public void get(K key) { // с помощью этого метода мы получаем пару ключ-значение
        int index = hash(key);
        if (table[index] == null) {
            table[index] = new LinkedList< Entry< K, V> > ();
        }
        for (Entry<K> entry : table[index]) {
            if (entry.getKey().equals(key)) {
                entry.getValue(value);
                return value;
            }
        }
    }
    public void remove(K key) { // с помощью этого метода мы удаляем пару ключ-значение
        int index = hash(key);
        if (table[index] == null) {
            table[index] = new LinkedList< Entry< K, V> > ();
        }
        for (Entry< K, V> entry : table[index]) {
            if (entry.getKey().equals(key)) {
                entry.setValue(value);
                return;
            }
        }
        table[index].remove(key, value);
        size--;
    }
} */
import java.util.LinkedList;

public class MyHashTable<K, V> {
    private static class Entry<K, V> {
        K key;
        V value;
        Entry(K key, V value) {
            this.key = key;
            this.value = value;
        }
        public K getKey() { return key; }
        public V getValue() { return value; }
        public void setValue(V value) { this.value = value; }
    }

    private static final int DEFAULT_CAPACITY = 16;
    private LinkedList<Entry<K, V>>[] table;
    private int size = 0;

    @SuppressWarnings("unchecked")
    public MyHashTable() {
        table = (LinkedList<Entry<K, V>>[]) new LinkedList[DEFAULT_CAPACITY];
    }

    private int hash(K key) {
        return (key.hashCode() & 0x7fffffff) % table.length;
    }

    public void put(K key, V value) {
        int index = hash(key);
        if (table[index] == null) {
            table[index] = new LinkedList<>();
        }
        for (Entry<K, V> entry : table[index]) {
            if (entry.getKey().equals(key)) {
                entry.setValue(value);
                return;
            }
        }
        table[index].add(new Entry<>(key, value));
        size++;
    }

    public String get(K key){
        int index = hash(K);
        // проверка 
        for (Entry<K, V> entry: table[index]){
            if (entry.getKey.equals(key)){
                entry.getValue(value);
                return value;
            }
        }
    }

    public void remove(K key){
        int index = hash(key);
        // проверка
        for (Entry<K, V> entry : table[index]) {
            if (entry.getKey().equals(key)) {
                entry.getValue(value);
                table[index].remove(Entry(key, value));
                size--;
                return;
            }
        }
        
    }
    public static void main(String[] args){
            MyHashTable<String, Integer> ht = new MyHashTable<>();
            ht.put("One ", 1);
            ht.put("Two ", 2);
            ht.put("Three ", 3);
            //System.out.println("Hashtable Elements: " + ht);
            ht.get("One ");
            //System.out.println(ht.get("One "));
        }

}