package org.example.spring_lab3_notifications.controller;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.example.spring_lab3_notifications.model.dto.NotificationDto;
import org.example.spring_lab3_notifications.model.entity.Notification;
import org.example.spring_lab3_notifications.model.enums.NotificationChannel;
import org.example.spring_lab3_notifications.model.enums.NotificationStatus;
import org.example.spring_lab3_notifications.service.NotificationService;
import org.springframework.web.bind.annotation.*;
import java.util.List;
@RestController
@RequestMapping("/notifications")
@RequiredArgsConstructor
public class NotificationController {
    private final NotificationService notificationService;
    public NotificationDto mapToDto(Notification response){
        return NotificationDto.builder()
                .title(response.getTitle())
                .message(response.getMessage())
                .channel(response.getChannel())
                .status(response.getStatus())
                .createdAt(response.getCreatedAt())
                .sentAt(response.getSentAt())
                .recipientId(response.getRecipient().getId())
                .build();
    }
    @PostMapping("/add")
    public NotificationDto createNotification(@RequestBody @Valid
                                              NotificationDto request) {
        Notification response =
                notificationService.createNotification(request);
        return mapToDto(response);
    }
    @GetMapping("/all")
    public List<NotificationDto> getAllNotifications() {
        return notificationService.getAllNotifications().stream()
                .map(this::mapToDto)
                .toList();

    }
    @GetMapping("/{id}")
    public NotificationDto getNotificationById(@PathVariable Long id) {
        Notification response = notificationService.getNotificationById(id);
        return mapToDto(response);
    }
    @PutMapping("/{id}/{status}")
    public NotificationDto updateStatus(@PathVariable Long id,
                                        @PathVariable NotificationStatus status) {
        Notification response = notificationService.updateNotificationStatus(id, status);
        return mapToDto(response);
    }
    @PutMapping("/{id}")
    public NotificationDto updateNotification(@PathVariable Long id,
                                              @RequestBody @Valid
                                              NotificationDto request) {
        Notification response = notificationService.updateNotification(id,
                request);
        return mapToDto(response);
    }
    @DeleteMapping("/{id}")
    public String deleteNotification(@PathVariable Long id) {
        notificationService.deleteNotification(id);
        return "Уведомление удалено";
    }
    @GetMapping("/status/{status}")
    public List<NotificationDto> getByStatus(@PathVariable
                                             NotificationStatus status) {
        return notificationService.getNotificationsByStatus(status).stream()
                .map(this::mapToDto)
                .toList();
    }
    @GetMapping("/channel/{channel}")
    public List<NotificationDto> getByChannel(@PathVariable
                                              NotificationChannel channel) {
        return
                notificationService.getNotificationsByChannel(channel).stream()
                        .map(this::mapToDto)
                        .toList();
    }
    @GetMapping("/recipient/{recipientId}")
    public List<NotificationDto> getByRecipientId(@PathVariable Long
                                                          recipientId) {
        return
                notificationService.getNotificationsByRecipientId(recipientId).stream()
                        .map(this::mapToDto)
                        .toList();
    }
    @GetMapping("/status/channel/{status}/{channel}")
    public List<NotificationDto> getByStatusAndChannel(@PathVariable NotificationStatus status, @PathVariable NotificationChannel channel) {
        return
                notificationService.getNotificationByStatusAndChannel(status, channel).stream()
                        .map(this::mapToDto)
                        .toList();
    }
    @GetMapping("/sort/date")
    public List<Notification> getNotificationSortedByDate() {
        return notificationService.getNotificationSortedByDate();
    }
    @GetMapping("/recipient/status/{recipient}/{status}")
    public List<NotificationDto> getByRecipientIdAndStatus(@PathVariable Long recipient, @PathVariable NotificationStatus status){
        return
                notificationService.getNotificationsByRecipientIdAndStatus(recipient, status).stream()
                        .map(this::mapToDto)
                        .toList();
    }

}