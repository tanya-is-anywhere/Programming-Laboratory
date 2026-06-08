package org.example.service;
import org.example.spring_lab3_notifications.model.dto.NotificationDto;
import org.example.spring_lab3_notifications.model.entity.Notification;
import org.example.spring_lab3_notifications.model.entity.User;
import org.example.spring_lab3_notifications.model.enums.NotificationChannel;
import org.example.spring_lab3_notifications.model.enums.NotificationStatus;
import org.example.spring_lab3_notifications.repository.NotificationRepository;
import org.example.spring_lab3_notifications.repository.UserRepository;
import org.example.spring_lab3_notifications.service.NotificationService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.time.LocalDateTime;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
@ExtendWith(MockitoExtension.class)
class NotificationServiceTest {
    @Mock
    private NotificationRepository notificationRepository;
    @Mock
    private UserRepository userRepository;
    @InjectMocks
    private NotificationService notificationService;
    @Test
    void shouldCreateNotification() {
        User user = new User();
        user.setId(1L);
        user.setEmail("ivan@example.com");
        NotificationDto dto = NotificationDto.builder()
                .title("Напоминание")
                .message("Завтра пара по Spring")
                .channel(NotificationChannel.EMAIL)
                .recipientId(1L)
                .build();
        Notification savedNotification = new Notification();
        savedNotification.setTitle(dto.getTitle());
        savedNotification.setMessage(dto.getMessage());
        savedNotification.setChannel(dto.getChannel());
        savedNotification.setRecipient(user);
        when(userRepository.findById(1L)).thenReturn(Optional.of(user));
        when(notificationRepository.save(any(Notification.class))).thenReturn(savedNotification);
        Notification result = notificationService.createNotification(dto);
        assertNotNull(result);
        assertEquals("Напоминание", result.getTitle());
        assertEquals(NotificationChannel.EMAIL, result.getChannel());
        assertEquals(user, result.getRecipient());
    }
    @Test
    void shouldGetNotificationById() {
        Long notificationId = 1L;
        User user = new User();
        user.setId(1L);
        user.setName("Иван Иванов");
        user.setEmail("ivan@example.com");
        Notification mockNotification = new Notification();
        mockNotification.setId(notificationId);
        mockNotification.setTitle("Напоминание");
        mockNotification.setMessage("Завтра дедлайн по Spring");
        mockNotification.setChannel(NotificationChannel.EMAIL);
        mockNotification.setStatus(NotificationStatus.CREATED);
        mockNotification.setRecipient(user);
        mockNotification.setCreatedAt(LocalDateTime.now());
        when(notificationRepository.findById(notificationId)).thenReturn(Optional.of(mockNotification));
        Notification result = notificationService.getNotificationById(notificationId);
        assertNotNull(result);
        assertEquals(notificationId, result.getId());
        assertEquals("Напоминание", result.getTitle());
        assertEquals("Завтра дедлайн по Spring", result.getMessage());
        assertEquals(NotificationChannel.EMAIL, result.getChannel());
        assertEquals(NotificationStatus.CREATED, result.getStatus());
        assertNotNull(result.getRecipient());
        assertEquals("Иван Иванов", result.getRecipient().getName());
        assertEquals("ivan@example.com", result.getRecipient().getEmail());
        verify(notificationRepository).findById(notificationId);
    }
    @Test
    void shouldThrowExceptionWhenNotificationNotFound() {
        Long notificationId = 1L;
        when(notificationRepository.findById(notificationId)).thenReturn(Optional.empty());
        assertThrows(RuntimeException.class, () -> {
            notificationService.getNotificationById(notificationId);
        });
        verify(notificationRepository).findById(notificationId);
    }
}
