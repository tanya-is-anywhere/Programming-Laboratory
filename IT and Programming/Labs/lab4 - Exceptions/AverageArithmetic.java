public class AverageArithmetic {
    public static double calculateAverage(Object[] nums){
        int lengthM = nums.length;
        int sum = 0;
        try {
            for (int i = 0; i < lengthM; i++) {
            sum += (int) nums[i];
            
            }
            return sum/lengthM;
            
        } catch (ArrayIndexOutOfBoundsException e) {
            System.out.println(e);
            System.out.println("Произошла ошибка: индекс массива выходит за границы.");
            return 0;
        } catch (ClassCastException e) {
            System.out.println(e);
            System.out.println("Произошла ошибка: вы присвоили элементу массива значения неверного типа.");
            return 0;
        }
        
        
    }
    public static void main(String[] args) {
        
        Object[] numsNew = {10, 78, 112, "hfe"};
        Double average = calculateAverage(numsNew);
        System.out.println("Среднее арифметическое: " + average);

    }
}