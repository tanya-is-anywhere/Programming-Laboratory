package org.example.spring_lab3_notifications.service;
import lombok.RequiredArgsConstructor;
import org.example.spring_lab3_notifications.model.dto.NotificationDto;
import org.example.spring_lab3_notifications.model.entity.Notification;
import org.example.spring_lab3_notifications.model.entity.User;
import org.example.spring_lab3_notifications.model.enums.NotificationChannel;
import org.example.spring_lab3_notifications.model.enums.NotificationStatus;
import org.example.spring_lab3_notifications.repository.NotificationRepository;
import org.example.spring_lab3_notifications.repository.UserRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
@Service
@RequiredArgsConstructor
public class NotificationService {
    private final NotificationRepository notificationRepository;
    private final UserRepository userRepository;
    @Transactional
    public Notification createNotification(NotificationDto request) {
        /*if (true) {
            throw new RuntimeException("Искусственная ошибка");
        }*/
        User user =
                userRepository.findById(request.getRecipientId()).orElseThrow();
        Notification notification = new Notification();
        notification.setTitle(request.getTitle());
        notification.setMessage(request.getMessage());
        notification.setChannel(request.getChannel());
        notification.setStatus(NotificationStatus.CREATED);
        notification.setCreatedAt(LocalDateTime.now());
        notification.setRecipient(user);
        notificationRepository.save(notification);


        return notification;
    }
    public List<Notification> getAllNotifications() {
        return notificationRepository.findAll();
    }
    public Notification getNotificationById(Long id) {
        return notificationRepository.findById(id).orElseThrow();
    }
    public List<Notification> getNotificationByStatusAndChannel(NotificationStatus status, NotificationChannel channel) {
        return notificationRepository.findByStatusAndChannelCustom(status, channel);
    }
    public List<Notification> getNotificationSortedByDate() {
        return notificationRepository.sortedNotificationsByDate();
    }
    public List<Notification> getNotificationsByRecipientIdAndStatus(Long recipientId, NotificationStatus status) {
        return notificationRepository.findNotificationsByRecipientIdAndStatus(recipientId, status);
    }
    public Notification updateNotificationStatus(Long id, NotificationStatus newStatus) {
        Notification notification = notificationRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Notification not found with id: " + id));

        notification.setStatus(newStatus);
        if (newStatus == NotificationStatus.SENT) {
            notification.setSentAt(LocalDateTime.now());
        }
        return notificationRepository.save(notification);
    }
    public Notification updateNotification(Long id, NotificationDto request)
    {
        Notification notification =
                notificationRepository.findById(id).orElseThrow();
        notification.setTitle(request.getTitle());
        notification.setMessage(request.getMessage());
        notification.setChannel(request.getChannel());
        notification.setStatus(request.getStatus());
        return notificationRepository.save(notification);
    }
    public void deleteNotification(Long id) {
        Notification notification =
                notificationRepository.findById(id).orElseThrow();
        notificationRepository.delete(notification);
    }
    public List<Notification> getNotificationsByStatus(NotificationStatus
                                                               status) {
        return notificationRepository.findByStatus(status);
    }
    public List<Notification> getNotificationsByChannel(NotificationChannel
                                                                channel) {
        return notificationRepository.findByChannel(channel);
    }
    public List<Notification> getNotificationsByRecipientId(Long
                                                                    recipientId) {
        return notificationRepository.findByRecipientId(recipientId);
    }
}