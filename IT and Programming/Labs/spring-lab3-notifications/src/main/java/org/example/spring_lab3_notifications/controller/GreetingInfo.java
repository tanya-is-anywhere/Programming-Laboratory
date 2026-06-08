package org.example.spring_lab3_notifications.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/greetinfo")
public class GreetingInfo {

    @GetMapping
    public String greet(@RequestParam(name = "name", required = false) String name, @RequestParam(name = "age", required = false) String age) {
        if (name == null || name.trim().isEmpty()) {
            return "Привет, незнакомец!";
        }
        return "Имя:" + name + ", возраст: " + age;
    }
}
