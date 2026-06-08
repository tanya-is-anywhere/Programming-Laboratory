package org.example;

import org.junit.jupiter.api.*;
public class LifecycleTest {
    @BeforeEach
    void setUp() {
        System.out.println("Подготовка перед каждым тестом");
    }
    @AfterEach
    void tearDown() {
        System.out.println("Завершение после каждого теста");
    }
    @Test
    void firstTest() {
        System.out.println("Первый тест");
    }
    @Test
    void secondTest() {
        System.out.println("Второй тест");
    }
}
