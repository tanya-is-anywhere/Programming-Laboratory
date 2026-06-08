package org.example.spring_lab3_notifications.service;

import org.springframework.stereotype.Service;

@Service
public class PushService implements MessageService {

    @Override
    public void sendMessage(String message, String recipient) {
        System.out.println("PUSH-уведомление для " + recipient + ": " + message);
    }
}
