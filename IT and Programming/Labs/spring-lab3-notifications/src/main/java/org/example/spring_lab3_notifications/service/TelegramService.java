package org.example.spring_lab3_notifications.service;

import org.springframework.context.annotation.Primary;
import org.springframework.stereotype.Service;

@Service
public class TelegramService implements MessageService {
    @Override
    public void sendMessage ( String message , String recipient ) {
        System.out.println("Telegram sends message to " + recipient + ": " + message);
    }
}