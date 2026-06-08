package org.example;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Spy;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class SimpleTest {

    // Существующий тест
    @Test
    void shouldAddNumbers() {
        int result = 2 + 3;
        assertEquals(5, result);
    }

    @Test
    void shouldVerifyAddMethodCallWithSpy() {
        List<String> list = new ArrayList<>();
        List<String> spyList = spy(list);

        spyList.add("Spring");
        spyList.add("Boot");

        verify(spyList, times(1)).add("Spring");
        verify(spyList, times(1)).add("Boot");
        verify(spyList, times(2)).add(anyString());

        assertEquals(2, spyList.size());
    }

    @Spy
    private List<String> annotatedSpyList = new ArrayList<>();
    @Test
    void shouldDemonstrateAnnotationSpy() {
        annotatedSpyList.add("Annotation");
        annotatedSpyList.add("Spy");
        annotatedSpyList.add("Test");

        verify(annotatedSpyList, times(1)).add("Annotation");
        verify(annotatedSpyList, times(1)).add("Spy");
        verify(annotatedSpyList, times(1)).add("Test");
        verify(annotatedSpyList, times(3)).add(anyString());

        assertEquals(3, annotatedSpyList.size());
        assertEquals("Annotation", annotatedSpyList.get(0));
        assertEquals("Spy", annotatedSpyList.get(1));
        assertEquals("Test", annotatedSpyList.get(2));

        when(annotatedSpyList.size()).thenReturn(100);
        assertEquals(100, annotatedSpyList.size());
    }
}
