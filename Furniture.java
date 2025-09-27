public class Furniture {
    abstract static class StoreFurniture{ //абстракция
        abstract void Info();

        public double getCost(){
            return 0;
        }
    }

    static class Chair extends StoreFurniture{
        private double height;
        private double seatWidth;
        private double seatDepth;
        private double tiltBackrest;
        private double angleSeatBackrest;
        private double cost;

        public Chair(double height, double seatWidth, double seatDepth, double tiltBackrest, double angleSeatBackrest, double cost){
            this.height = height;
            this.seatWidth = seatWidth;
            this.seatDepth = seatDepth;
            this.tiltBackrest = tiltBackrest;
            this.angleSeatBackrest = angleSeatBackrest;
            this.cost = cost;
        }

        @Override
        void Info(){
            System.out.println("Стул: информация");
            System.out.println("Высота: " + this.height);
            System.out.println("Ширина сидения: " +  this.seatWidth);
            System.out.println("Глубина сидения: " + this.seatDepth);
            System.out.println("Наклон спинки: " + this.tiltBackrest);
            System.out.println("Угол между сиденьем и спинкой: " + this.angleSeatBackrest);
            System.out.println("Цена: " + this.cost);

        }
        public void changeCost(double cost){ // сеттер
            this.cost = cost;
        }
        public double getCost(){ // геттер
            return this.cost;
        }

    }
    static class Table extends StoreFurniture{
        private double height;
        private double width;
        private double depth;
        private double cost;
        
        public Table(double height, double width, double depth, double cost){
            this.height = height;
            this.width = width;
            this.depth = depth;
            this.cost = cost;
            
        }

        @Override
        void Info(){
            System.out.println("Стол: информация");
            System.out.println("Высота: " + this.height);
            System.out.println("Ширина: " + this.width);
            System.out.println("Глубина: " + this.depth);

        }
        public void changeCost(double cost){ // сеттер
            this.cost = cost;
        }
        public double getCost(){ // геттер
            return this.cost;
        }

    }
    static class Bed extends StoreFurniture{ // наследование
        private String type; // инкапсуляция
        private String design;
        private String material;
        private double cost;
        private static int counter;
        
        public Bed(String type, String design, String material, double cost){ // конструктор
            this.type = type;
            this.design = design;
            this.material = material;
            this.cost = cost;
            counter++;
            
        }


        @Override
        void Info(){ // динамический полиморфизм
            System.out.println("Кровать: информация");
            System.out.println("Тип: " + this.type);
            System.out.println("Дизайн: " + this.design);
            System.out.println("Материал: " + this.material);
            System.out.println("Цена: " + this.cost);

        }

        public void changeCost(double cost){ // сеттер
            this.cost = cost;
        }

        public double getCost(){ // геттер
            return this.cost;
        }

        public static int getCount(){ // вывод числа объектов
            return counter;
        }


    }
    public static void main(String[] args){
            Bed bed1 = new Bed ("двухспальная","Классический","Дерево",35000);
            Bed bed2 = new Bed ("односпальная","Лофт","ЛДСП",15000);
            bed1.Info();
            bed2.Info();
            System.out.println("Количество кроватей: " + Bed.getCount());
            bed1.changeCost(37000);
            System.out.println("Новая цена кровати: " + bed1.cost);
            Chair chair1 = new Chair(123, 35, 40, 65, 65, 999.99);
            chair1.Info();
        }
    
    
}
