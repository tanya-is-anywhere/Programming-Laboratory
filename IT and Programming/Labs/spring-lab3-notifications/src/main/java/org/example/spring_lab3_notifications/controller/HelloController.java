package org.example.spring_lab3_notifications.controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
@RestController
public class HelloController {
    @GetMapping ("/hello")
    public String sayHello () {
        return "Привет, Spring Boot!";
        }
    @GetMapping ("/goodbye")
    public String sayGoodbye () {
        return "До свидания, Spring Boot!";
    }
}
