package org.example.spring_lab3_notifications.controller;

import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.example.spring_lab3_notifications.mapper.UserMapper;
import org.example.spring_lab3_notifications.model.dto.UserDto;
import org.example.spring_lab3_notifications.service.UserService;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/users")
@RequiredArgsConstructor
public class UserController {

    private final UserService userService;
    private final UserMapper userMapper;

    @GetMapping("/all")
    public List<UserDto> getAllUsers() {
        return userService.getAllUsers().stream()
                .map(userMapper::toDto)
                .toList();
    }

    @PostMapping("/add")
    public UserDto createUser(@RequestBody @Valid UserDto request) {
        return userMapper.toDto(userService.createUser(request));
    }

    @GetMapping("/{id}")
    public UserDto getUserById(@PathVariable Long id) {
        return userMapper.toDto(userService.getUserById(id));
    }

    @PutMapping("/{id}")
    public UserDto updateUser(@PathVariable Long id, @RequestBody @Valid UserDto request) {
        return userMapper.toDto(userService.updateUser(id, request));
    }

    @PreAuthorize("hasRole('ADMIN')")
    @DeleteMapping("/{id}")
    public String deleteUser(@PathVariable Long id) {
        userService.deleteUser(id);
        return String.format("Пользователь %s удалён", id);
    }
}
