# Java

Java is an object-oriented programming language where code is organized into classes and objects.

## Key Features

- Object-oriented
- Easier to read than some older languages
- Good performance
- Can multitask easily
- Platform-independent

## Syntax

Java code is usually written inside classes. For example,

```java
public class Main {
    
}
```

To print text into the console in Java, you write `System.out.println(text)`. For example, 
```java
public class Main {
    public static void main(String[] args) {
        System.out.println('Hello World!');
    }
}
```
prints:
```text
Hello World!
```
To create text variables, you write `String var = 'Hello World!'`. Full code:
```java
public class Main {
    public static void main(String[] args) {
        String var = 'Hello World!'
        System.out.println(var);
    }
}
```
which also prints:
```text
Hello World!
```

You can also change the variable:
```java
String var = "Hello World!"
System.out.println(var)
var = "Hello!"
System.out.println(var)
```
but make sure it's initialized first

The code above prints:
```text
Hello World!
Hello!
```

Even though the `System.out.println()` didn't change, the output changed because of the variable.

## Frameworks

- Spring Boot: backend and web application development
- Spring MVC: web application architecture
- Hibernate: object-relational mapping and database access
- Jakarta EE: enterprise Java applications
- LibGDX: game development
