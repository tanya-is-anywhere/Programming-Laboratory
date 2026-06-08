package org.example.spring_lab3_notifications;
import org.example.spring_lab3_notifications.config.AnotherConfig;
import org.example.spring_lab3_notifications.config.AppConfig ;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Import ;

@SpringBootApplication
@Import(AppConfig.class)
public class SpringLab3NotificationsApplication {
	public static void main(String[] args) {
		SpringApplication.run(SpringLab3NotificationsApplication.class,args);
	}

}
