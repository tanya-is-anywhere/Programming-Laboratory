package org.example.service;
import org.example.spring_lab3_notifications.model.dto.NotificationDto;
import org.example.spring_lab3_notifications.model.enums.NotificationChannel;
import org.example.spring_lab3_notifications.repository.NotificationRepository;
import org.example.spring_lab3_notifications.repository.UserRepository;
import org.example.spring_lab3_notifications.service.NotificationService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Optional;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.Mockito.when;
@ExtendWith(MockitoExtension.class)
class NotificationServiceExceptionTest {
    @Mock
    private NotificationRepository notificationRepository;
    @Mock
    private UserRepository userRepository;
    @InjectMocks
    private NotificationService notificationService;
    @Test
    void shouldThrowExceptionWhenUserNotFound() {
        NotificationDto dto = NotificationDto.builder()
                .title("Напоминание")
                .message("Сообщение")
                .channel(NotificationChannel.EMAIL)
                .recipientId(99L)
                .build();
        when(userRepository.findById(99L)).thenReturn(Optional.empty());
        assertThrows(RuntimeException.class, () ->
                notificationService.createNotification(dto));
    }
}
