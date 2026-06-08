package org.example;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertThrows;
public class ExceptionTest {
    @Test
    void shouldThrowException() {
        assertThrows(ArithmeticException.class, () -> {
            int result = 10 / 0;
        });
    }
}
