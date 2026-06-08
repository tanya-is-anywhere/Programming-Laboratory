package org.example.spring_lab3_notifications.mapper;

import org.example.spring_lab3_notifications.model.dto.UserDto;
import org.example.spring_lab3_notifications.model.entity.User;
import org.springframework.stereotype.Component;

@Component
public class UserMapper {

    public UserDto toDto(User user) {
        if (user == null) return null;
        return UserDto.builder()
                .name(user.getName())
                .email(user.getEmail())
                .phone(user.getPhone())
                .telegramChatId(user.getTelegramChatId())
                .deviceToken(user.getDeviceToken())
                .createdAt(user.getCreatedAt())
                .build();
    }

}