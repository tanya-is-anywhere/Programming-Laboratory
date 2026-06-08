package org.example.spring_lab3_notifications.config;

import org.example.spring_lab3_notifications.service.PushService;
import org.springframework.context.annotation.Bean;

public class AnotherConfig {
    @Bean
    public PushService pushService () {
        return new PushService () ;
    }

}
