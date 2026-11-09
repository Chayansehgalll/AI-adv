public class DemoApp {
    public static void main(String[] args) {
        String[] names = {"Alice", "Bob", "Charlie"};
        printNames(names);
    }

    public static void printNames(String[] names) {
        for (int i = 0; i < names.length; i++) {
            System.out.println((i + 1) + " " + names[i]);
        }
    }
}
