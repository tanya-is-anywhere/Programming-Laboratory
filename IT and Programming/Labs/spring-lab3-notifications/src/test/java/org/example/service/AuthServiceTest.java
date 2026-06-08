package org.example.service;
import org.example.spring_lab3_notifications.model.dto.RegisterRequest;
import org.example.spring_lab3_notifications.model.entity.User;
import org.example.spring_lab3_notifications.model.enums.UserRole;
import org.example.spring_lab3_notifications.repository.UserRepository;
import org.example.spring_lab3_notifications.service.AuthService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.security.crypto.password.PasswordEncoder;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class AuthServiceTest {

    @Mock
    private UserRepository userRepository;

    @Mock
    private PasswordEncoder passwordEncoder;

    @InjectMocks
    private AuthService authService;

    @Test
    void shouldRegisterUserSuccessfully() {
        RegisterRequest request = new RegisterRequest();
        request.setName("Иван Иванов");
        request.setEmail("ivan@example.com");
        request.setPassword("qwerty123");

        when(userRepository.findByEmail(request.getEmail())).thenReturn(Optional.empty());
        when(passwordEncoder.encode(request.getPassword())).thenReturn("encodedPassword123");

        authService.register(request);

        verify(userRepository).findByEmail(request.getEmail());
        verify(passwordEncoder).encode(request.getPassword());
        verify(userRepository).save(any(User.class));
    }

    @Test
    void shouldThrowExceptionWhenEmailAlreadyExists() {
        RegisterRequest request = new RegisterRequest();
        request.setName("Иван Иванов");
        request.setEmail("existing@example.com");
        request.setPassword("qwerty123");

        User existingUser = new User();
        existingUser.setEmail("existing@example.com");

        when(userRepository.findByEmail(request.getEmail())).thenReturn(Optional.of(existingUser));
        assertThrows(RuntimeException.class, () -> authService.register(request));

        verify(userRepository).findByEmail(request.getEmail());
        verify(userRepository, never()).save(any(User.class));
        verify(passwordEncoder, never()).encode(anyString());
    }

    @Test
    void shouldRegisterAdminSuccessfully() {
        RegisterRequest request = new RegisterRequest();
        request.setName("Admin User");
        request.setEmail("admin@example.com");
        request.setPassword("admin123");

        when(userRepository.findByEmail(request.getEmail())).thenReturn(Optional.empty());
        when(passwordEncoder.encode(request.getPassword())).thenReturn("encodedAdminPassword");
        authService.registerAdmin(request);
        verify(userRepository).findByEmail(request.getEmail());
        verify(passwordEncoder).encode(request.getPassword());
        verify(userRepository).save(any(User.class));
    }

    @Test
    void shouldThrowExceptionWhenAdminEmailAlreadyExists() {
        RegisterRequest request = new RegisterRequest();
        request.setName("Admin User");
        request.setEmail("existing@example.com");
        request.setPassword("admin123");

        User existingUser = new User();
        existingUser.setEmail("existing@example.com");

        when(userRepository.findByEmail(request.getEmail())).thenReturn(Optional.of(existingUser));
        assertThrows(RuntimeException.class, () -> authService.registerAdmin(request));

        verify(userRepository).findByEmail(request.getEmail());
        verify(userRepository, never()).save(any(User.class));
    }
    @Test
    void shouldSetUserRoleWhenRegistering() {
        RegisterRequest request = new RegisterRequest();
        request.setName("Обычный Пользователь");
        request.setEmail("user@example.com");
        request.setPassword("user123");

        when(userRepository.findByEmail(request.getEmail())).thenReturn(Optional.empty());
        when(passwordEncoder.encode(request.getPassword())).thenReturn("encodedUserPassword");

        authService.register(request);

        verify(userRepository).save(argThat(user ->
                user.getRole() == UserRole.ROLE_USER
        ));
    }

    @Test
    void shouldSetAdminRoleWhenRegisteringAdmin() {
        RegisterRequest request = new RegisterRequest();
        request.setName("Администратор");
        request.setEmail("admin@example.com");
        request.setPassword("admin123");

        when(userRepository.findByEmail(request.getEmail())).thenReturn(Optional.empty());
        when(passwordEncoder.encode(request.getPassword())).thenReturn("encodedAdminPassword");

        authService.registerAdmin(request);

        verify(userRepository).save(argThat(user ->
                user.getRole() == UserRole.ROLE_ADMIN
        ));
    }
}
