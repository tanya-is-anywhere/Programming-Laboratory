class Program{
 
    public static void main(String[] args) {  
 
        Person tom = new Person(); // вызов конструктора по умолчанию
 
        //tom.name = "Tom";
        //tom.age = 41;
 
        tom.print();
    }
}
 
class Person{
 
    String name;
    int age;
 
    void print(){
 
        System.out.printf("Name: %s; Age: %d\n", name, age);
    }
}