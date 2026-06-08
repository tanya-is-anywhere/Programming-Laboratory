package org.example.spring_lab3_notifications.service;
import lombok.RequiredArgsConstructor;
import org.example.spring_lab3_notifications.model.dto.RegisterRequest;
import org.example.spring_lab3_notifications.model.entity.User;
import org.example.spring_lab3_notifications.model.enums.UserRole;
import org.example.spring_lab3_notifications.repository.UserRepository;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import java.time.LocalDateTime;
@Service
@RequiredArgsConstructor
public class AuthService {
    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;
    public void register(RegisterRequest request) {
        if (userRepository.findByEmail(request.getEmail()).isPresent()) {
            throw new RuntimeException("Email уже зарегистрирован");
        }
        User user = new User();
        user.setName(request.getName());
        user.setEmail(request.getEmail());
        user.setPassword(passwordEncoder.encode(request.getPassword()));
        user.setRole(UserRole.ROLE_USER);
        user.setCreatedAt(LocalDateTime.now());
        userRepository.save(user);

    }
    public void registerAdmin(RegisterRequest request) {
        if (userRepository.findByEmail(request.getEmail()).isPresent()) {
            throw new RuntimeException("Email уже зарегистрирован");
        }
        User user = new User();
        user.setName(request.getName());
        user.setEmail(request.getEmail());
        user.setPassword(passwordEncoder.encode(request.getPassword()));
        user.setRole(UserRole.ROLE_ADMIN);
        user.setCreatedAt(LocalDateTime.now());
        userRepository.save(user);
    }
}
