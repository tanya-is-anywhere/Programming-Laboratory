import java.util.*;
import java.time.LocalDate;

// Класс для представления заказа
class Order {
    private LocalDate orderDate;
    private List<String> products;
    private String status;

    public Order(LocalDate date, List<String> products, String status) {
        this.orderDate = date;
        this.products = products;
        this.status = status;
    }

    // Геттеры и сеттеры
    public LocalDate getOrderDate() {
        return orderDate;
    }

    public List<String> getProducts() {
        return products;
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    @Override
    public String toString() {
        return "Заказ от " + orderDate + 
               "\nСтатус: " + status +
               "\nТовары: " + products;
    }
}

// Класс для работы с заказами
public class OrderManager {
    private Map<Integer, Order> orders;

    public OrderManager() {
        // Создаем HashMap для хранения заказов
        orders = new HashMap<>();
    }

    // Добавление нового заказа
    public void addOrder(int orderId, Order order) {
        if (orders.containsKey(orderId)) {
            System.out.println("Заказ с таким ID уже существует");
        } else {
            orders.put(orderId, order);
            System.out.println("Заказ добавлен");
        }
    }

    // Поиск заказа по ID
    public Order findOrder(int orderId) {
        return orders.getOrDefault(orderId, null);
    }

    // Удаление заказа
    public void removeOrder(int orderId) {
        if (orders.remove(orderId) != null) {
            System.out.println("Заказ удален");
        } else {
            System.out.println("Заказ не найден");
        }
    }

    // Изменение статуса заказа
    public void updateOrderStatus(int orderId, String newStatus) {
        Order order = orders.get(orderId);
        if (order != null) {
            order.setStatus(newStatus);
            System.out.println("Статус заказа изменен");
        } else {
            System.out.println("Заказ не найден");
        }
    }

    // Пример использования
    public static void main(String[] args) {
        OrderManager manager = new OrderManager();

        // Создаем тестовый заказ
        List<String> products = Arrays.asList("Телефон", "Зарядное устройство");
        Order order = new Order(LocalDate.now(), products, "Ожидает обработки");

        // Добавляем заказ
        manager.addOrder(1, order);

        // Ищем заказ
        System.out.println("Найденный заказ: " + manager.findOrder(1));

        // Изменяем статус
        manager.updateOrderStatus(1, "Доставляется");

        // Удаляем заказ
        manager.removeOrder(1);
    }
}
