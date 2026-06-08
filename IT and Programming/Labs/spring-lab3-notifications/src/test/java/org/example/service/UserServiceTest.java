package org.example.service;
import org.example.spring_lab3_notifications.model.dto.UserDto;
import org.example.spring_lab3_notifications.model.entity.User;
import org.example.spring_lab3_notifications.repository.UserRepository;
import org.example.spring_lab3_notifications.service.UserService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class UserServiceTest {
    @Mock
    private UserRepository userRepository;
    @InjectMocks
    private UserService userService;
    @Test
    void shouldCreateUser() {
        UserDto dto = UserDto.builder()
                .name("Иван Иванов")
                .email("ivan@example.com")
                .phone("+79990001122")
                .deviceToken("device-token-123")
                .telegramChatId("123456789")
                .build();
        User savedUser = new User();
        savedUser.setName(dto.getName());
        savedUser.setEmail(dto.getEmail());
        savedUser.setPhone(dto.getPhone());
        savedUser.setDeviceToken(dto.getDeviceToken());
        savedUser.setTelegramChatId(dto.getTelegramChatId());
        when(userRepository.save(any(User.class))).thenReturn(savedUser);
        User result = userService.createUser(dto);
        assertNotNull(result);
        assertEquals("Иван Иванов", result.getName());
        assertEquals("ivan@example.com", result.getEmail());
    }
    @Test
    void shouldGetUserById() {
        Long userId = 1L;
        User mockUser = new User();
        mockUser.setId(userId);
        mockUser.setName("Иван Иванов");

        when(userRepository.findById(userId)).thenReturn(Optional.of(mockUser));

        User result = userService.getUserById(userId);

        assertNotNull(result);
        assertEquals(userId, result.getId());
    }
    @Test
    void shouldDeleteUser() {
        Long userId = 1L;
        User existingUser = new User();
        existingUser.setId(userId);
        existingUser.setName("Иван Иванов");
        existingUser.setEmail("ivan@example.com");

        when(userRepository.findById(userId)).thenReturn(Optional.of(existingUser));

        doNothing().when(userRepository).delete(existingUser);

        userService.deleteUser(userId);

        verify(userRepository, times(1)).findById(userId);
        verify(userRepository, times(1)).delete(existingUser);
    }
}

