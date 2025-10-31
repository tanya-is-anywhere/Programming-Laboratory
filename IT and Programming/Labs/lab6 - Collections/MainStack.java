
class Stack<T> {
    private T[] data;
    private int size;
    private final int capacity;

    public Stack(int capacity){        
        this.capacity = capacity;
        this.data = (T[]) new Object[capacity];
        this.size = 0;
    }
    public void push(T element){
        data[size++] = element;
    }
    public T pop(){
        T result = data[size-1];
        data[size-1] = null;
        size--;
        return result;
    }
    public T peek(){
        return data[size-1];
    }
    
}

public class MainStack {

    public static void main(String[] args) {
        Stack<Integer> stack = new Stack<> (10);
        stack.push(5);
        stack.push(3);
        stack.push(2);
        stack.push(3);
        stack.push(2);
        stack.push(3);
        System.out.println(stack.pop());
        System.out.println(stack.peek());
        stack.push(4);
        System.out.println(stack.pop());
        stack.push(1);
    }
}