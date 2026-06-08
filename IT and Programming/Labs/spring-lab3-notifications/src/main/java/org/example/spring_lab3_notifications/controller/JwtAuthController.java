package org.example.spring_lab3_notifications.controller;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.example.spring_lab3_notifications.model.dto.LoginRequest;
import org.example.spring_lab3_notifications.security.JwtService;
import org.springframework.http.ResponseEntity;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.BadCredentialsException;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/auth")
@RequiredArgsConstructor
public class JwtAuthController {
    private final AuthenticationManager authenticationManager;
    private final JwtService jwtService;
    @PostMapping("/login")
    public ResponseEntity<Map<String, String>> login(@RequestBody @Valid LoginRequest request) {
        try {
            authenticationManager.authenticate(
                    new UsernamePasswordAuthenticationToken(
                            request.getEmail(),
                            request.getPassword()
                    )
            );
            String token = jwtService.generateToken(request.getEmail());
            return ResponseEntity.ok(Map.of("token", token));
        } catch (BadCredentialsException e) {
            return ResponseEntity.status(401).body(Map.of("error", "Неверный email или пароль"));
        }
    }


}
