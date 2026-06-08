package org.example.spring_lab3_notifications.repository;
import org.example.spring_lab3_notifications.model.entity.Notification;
import org.example.spring_lab3_notifications.model.enums.NotificationChannel;
import org.example.spring_lab3_notifications.model.enums.NotificationStatus;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;
import java.util.List;
@Repository
public interface NotificationRepository extends JpaRepository<Notification,
        Long> {
    List<Notification> findByStatus(NotificationStatus status);

    List<Notification> findByChannel(NotificationChannel channel);

    List<Notification> findByRecipientId(Long recipientId);


    @Query("""
            select n
from Notification n
where n.status = :status
 and n.channel = :channel
""")
    List<Notification> findByStatusAndChannelCustom(@Param("status")
                                                    NotificationStatus status,

                                                    @Param("channel")
                                                    NotificationChannel channel);
    @Query(value = """
            select *
from notifications
where status = :status
 and channel = :channel
""", nativeQuery = true)
    List<Notification> findNativeByStatusAndChannel(@Param("status") String
                                                            status,
                                                    @Param("channel") String
                                                            channel);


@Query("SELECT n FROM Notification n ORDER BY n.createdAt DESC")
    List<Notification> sortedNotificationsByDate();

@Query(value = """
select n from Notification n where n.recipient.id = :recipientId
         and n.status = :status""")
    List<Notification> findNotificationsByRecipientIdAndStatus(@Param("recipientId") Long
                                                                       recipientId,
                                                               @Param("status") NotificationStatus
                                                                       status);
}

/* Данные списки были даны для реализации по лабораторной работе, но я создала их же, но с другими названиями.

    List<Notification> findByStatusAndChannel(NotificationStatus status,
                                              NotificationChannel channel);

    List<Notification> findByStatusOrderByCreatedAtAsc(NotificationStatus
                                                               status);

 */