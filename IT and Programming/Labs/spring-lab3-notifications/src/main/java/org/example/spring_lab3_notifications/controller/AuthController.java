package org.example.spring_lab3_notifications.controller;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.example.spring_lab3_notifications.model.dto.RegisterRequest;
import org.example.spring_lab3_notifications.service.AuthService;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
@RestController
@RequestMapping("/auth")
@RequiredArgsConstructor
public class AuthController {
    private final AuthService authService;
    @PostMapping("/register")
    public String register(@RequestBody @Valid RegisterRequest request) {
        System.out.println(">>> ЗАПРОС ДОШЁЛ ДО КОНТРОЛЛЕРА: " + request.getEmail());
        authService.register(request);
        return "Пользователь успешно зарегистрирован";
    }
    @PostMapping("/register/admin")
    @PreAuthorize("hasRole('ADMIN')")
    public String registerAdmin(@RequestBody @Valid RegisterRequest request) {
        authService.registerAdmin(request);
        return "Администратор успешно зарегистрирован";
    }
}
