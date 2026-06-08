package org.example.controller;

import org.example.spring_lab3_notifications.SpringLab3NotificationsApplication;
import org.example.spring_lab3_notifications.config.SecurityConfig;
import org.example.spring_lab3_notifications.model.entity.Notification;
import org.example.spring_lab3_notifications.model.entity.User;
import org.example.spring_lab3_notifications.model.enums.NotificationChannel;
import org.example.spring_lab3_notifications.model.enums.NotificationStatus;
import org.example.spring_lab3_notifications.service.NotificationService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.context.annotation.Import;
import org.springframework.security.test.context.support.WithMockUser;
import org.springframework.test.web.servlet.MockMvc;

import java.util.Arrays;
import java.util.List;

import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@SpringBootTest(classes = SpringLab3NotificationsApplication.class)
@Import(SecurityConfig.class)
@AutoConfigureMockMvc
class NotificationControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @MockBean
    private NotificationService notificationService;

    @Test
    @WithMockUser
    void shouldReturnAllNotifications() throws Exception {

        User recipient = new User();
        recipient.setId(1L);

        Notification notification1 = new Notification();
        notification1.setTitle("Title 1");
        notification1.setMessage("Message 1");
        notification1.setChannel(NotificationChannel.EMAIL);
        notification1.setStatus(NotificationStatus.CREATED);
        notification1.setRecipient(recipient);

        Notification notification2 = new Notification();
        notification2.setTitle("Title 2");
        notification2.setMessage("Message 2");
        notification2.setChannel(NotificationChannel.SMS);
        notification2.setStatus(NotificationStatus.SENT);
        notification2.setRecipient(recipient);

        List<Notification> notifications = Arrays.asList(notification1, notification2);

        when(notificationService.getAllNotifications()).thenReturn(notifications);

        String BASE_URL = "/notifications";
        mockMvc.perform(get(BASE_URL + "/all"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.length()").value(2))
                .andExpect(jsonPath("$[0].title").value("Title 1"));
    }
}
