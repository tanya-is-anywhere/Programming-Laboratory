import java.util.*;

// Класс, описывающий одну продажу
class Sale {
    String item;
    double price;

    public Sale(String item, double price) {
        this.item = item;
        this.price = price;
    }

    @Override
    public int hashCode() {
        return Objects.hash(item, price);
    }

    @Override
    public String toString() {
        return String.format("%s - %.2f руб.", item, price);
    }
}

// Основной класс для учёта продаж
public class SalesTracker {
    private Set<Sale> sales;

    public SalesTracker() {
        this.sales = new HashSet<>();
    }

    public void addSale(String item, double price) {
        Sale sale = new Sale(item, price);
        sales.add(sale);
        System.out.println("Добавлена продажа: " + sale);
    }

    public void printSales() {
        if (sales.isEmpty()) {
            System.out.println("Продажи отсутствуют.");
            return;
        }
        System.out.println("Список продаж:");
        for (Sale sale : sales) {
            System.out.println(sale);
        }
    }

    public double getTotalRevenue() {
        double total = 0;
        for (Sale sale : sales) {
            total += sale.price;
        }
        return total;
    }

    public String getMostPopularItem() {
        if (sales.isEmpty()) return "Нет продаж";

        Map<String, Integer> countMap = new HashMap<>();
        for (Sale sale : sales) {
            countMap.put(sale.item, countMap.getOrDefault(sale.item, 0) + 1);
        }

        String mostPopular = "";
        int maxCount = 0;
        for (Map.Entry<String, Integer> entry : countMap.entrySet()) {
            if (entry.getValue() > maxCount) {
                maxCount = entry.getValue();
                mostPopular = entry.getKey();
            }
        }
        return mostPopular + " (встречается " + maxCount + " раз)";
    }

    public static void main(String[] args) {
        SalesTracker tracker = new SalesTracker();

        // Добавляем продажи
        tracker.addSale("Яблоко", 50.0);
        tracker.addSale("Банан", 30.0);
        tracker.addSale("Яблоко", 55.0);  
        tracker.addSale("Груша", 40.0);
        tracker.addSale("Яблоко", 48.0);  

        // Выводим список продаж
        tracker.printSales();

        // Общая сумма
        System.out.printf("Общая сумма продаж: %.2f руб.%n", tracker.getTotalRevenue());

        // Самый популярный товар
        System.out.println("Самый популярный товар: " + tracker.getMostPopularItem());
    }
}